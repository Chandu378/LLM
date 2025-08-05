from fastapi import FastAPI, HTTPException
import json
import httpx
import tempfile
import os
from typing import List, Dict, Any
import google.generativeai as genai

# Import models first
from models import Clause, QueryRequest, QueryResponse
from pydantic import BaseModel

# Import services after models
from services import gemini_service, redis_service, pinecone_service
from config import config
from supabase_service import supabase_service
from supabase_db_service import supabase_db_service
from lightning_processor import lightning_processor

app = FastAPI(title="Document Query Assistant", version="1.0.0")

def analyze_document_context(clauses: List[Clause]) -> Dict[str, Any]:
    """Analyze document content to understand context and generate dynamic mappings"""
    if not clauses:
        return {"type": "general", "keywords": {}, "domain_terms": []}
    
    # Combine clause content for analysis
    combined_content = " ".join([clause.content.lower() for clause in clauses[:10]])
    
    # Detect document domain/type
    domain_indicators = {
        "insurance": ["policy", "premium", "coverage", "claim", "insured", "deductible", "exclusion"],
        "legal": ["contract", "agreement", "clause", "party", "liability", "jurisdiction", "breach"],
        "medical": ["patient", "treatment", "diagnosis", "medication", "doctor", "hospital", "therapy"],
        "financial": ["investment", "portfolio", "return", "risk", "asset", "liability", "equity"],
        "hr": ["employee", "salary", "benefits", "leave", "performance", "policy", "handbook"],
        "technical": ["system", "software", "hardware", "configuration", "protocol", "interface"],
        "academic": ["research", "study", "analysis", "methodology", "conclusion", "hypothesis"],
        "business": ["strategy", "market", "customer", "revenue", "profit", "operations"]
    }
    
    domain_scores = {}
    for domain, indicators in domain_indicators.items():
        score = sum(1 for indicator in indicators if indicator in combined_content)
        if score > 0:
            domain_scores[domain] = score
    
    # Determine primary domain
    primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "general"
    
    # Extract domain-specific terms from content
    import re
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_content)
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get most frequent domain-specific terms
    domain_terms = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]]
    
    return {
        "type": primary_domain,
        "domain_scores": domain_scores,
        "domain_terms": domain_terms
    }

def generate_dynamic_keyword_mappings(question: str, document_context: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate dynamic keyword mappings based on document context and question"""
    question_lower = question.lower()
    doc_type = document_context.get("type", "general")
    domain_terms = document_context.get("domain_terms", [])
    
    # Base mappings that work across domains
    base_mappings = {
        "what": ["definition", "meaning", "explanation", "description"],
        "how": ["process", "procedure", "method", "way", "steps"],
        "when": ["time", "period", "duration", "date", "schedule"],
        "where": ["location", "place", "address", "site"],
        "why": ["reason", "purpose", "cause", "rationale"],
        "who": ["person", "individual", "entity", "party"],
        "amount": ["cost", "price", "fee", "charge", "value", "sum"],
        "limit": ["maximum", "minimum", "cap", "threshold", "boundary"],
        "requirement": ["condition", "criteria", "prerequisite", "need"],
        "benefit": ["advantage", "coverage", "service", "feature"]
    }
    
    # Domain-specific mappings
    domain_mappings = {
        "insurance": {
            "coverage": ["protection", "benefit", "insurance", "policy", "plan"],
            "claim": ["reimbursement", "payment", "settlement", "compensation"],
            "premium": ["cost", "payment", "fee", "charge", "price"],
            "exclusion": ["not covered", "limitation", "restriction", "exception"],
            "waiting": ["period", "time", "duration", "delay"],
            "deductible": ["excess", "out-of-pocket", "copay", "contribution"]
        },
        "legal": {
            "contract": ["agreement", "document", "terms", "conditions"],
            "liability": ["responsibility", "obligation", "duty", "accountability"],
            "breach": ["violation", "default", "non-compliance", "failure"],
            "jurisdiction": ["court", "authority", "legal system", "venue"],
            "party": ["entity", "individual", "organization", "participant"]
        },
        "medical": {
            "treatment": ["therapy", "care", "procedure", "intervention"],
            "diagnosis": ["condition", "disease", "illness", "disorder"],
            "medication": ["drug", "medicine", "prescription", "pharmaceutical"],
            "patient": ["individual", "person", "client", "subject"],
            "doctor": ["physician", "practitioner", "healthcare provider"]
        },
        "financial": {
            "investment": ["asset", "security", "portfolio", "fund"],
            "return": ["profit", "gain", "yield", "income"],
            "risk": ["volatility", "uncertainty", "exposure", "hazard"],
            "equity": ["stock", "share", "ownership", "capital"]
        }
    }
    
    # Combine mappings
    final_mappings = base_mappings.copy()
    if doc_type in domain_mappings:
        final_mappings.update(domain_mappings[doc_type])
    
    # Add document-specific terms
    for term in domain_terms[:10]:  # Top 10 domain terms
        if term not in final_mappings:
            final_mappings[term] = [term]
    
    return final_mappings

def preprocess_question(question: str, document_context: Dict[str, Any] = None) -> str:
    """Advanced question preprocessing with dynamic context awareness"""
    if not document_context:
        document_context = {"type": "general", "domain_terms": []}
    
    question_lower = question.lower()
    
    # Generate dynamic keyword mappings
    keyword_mappings = generate_dynamic_keyword_mappings(question, document_context)
    
    # Apply semantic preprocessing
    processed_question = question
    
    # Extract key concepts from question
    import re
    question_words = re.findall(r'\b[a-zA-Z]{3,}\b', question_lower)
    
    # Find and apply best mappings
    for word in question_words:
        for key, synonyms in keyword_mappings.items():
            if word in synonyms or any(syn in word for syn in synonyms):
                # Replace with the most contextually appropriate term
                if key in document_context.get("domain_terms", []):
                    processed_question = processed_question.replace(word, key)
                break
    
    return processed_question

def generate_search_keywords(question: str, document_context: Dict[str, Any] = None) -> List[str]:
    """Generate comprehensive search keywords using semantic analysis and context"""
    if not document_context:
        document_context = {"type": "general", "domain_terms": []}
    
    question_lower = question.lower()
    keywords = []
    
    # Base question (always include)
    keywords.append(question)
    
    # Extract key concepts using NLP-like approach
    import re
    
    # Extract meaningful words (3+ characters, not common stop words)
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'way', 'will', 'with'}
    
    question_words = [word for word in re.findall(r'\b[a-zA-Z]{3,}\b', question_lower) 
                     if word not in stop_words]
    
    # Generate semantic variations for each key word
    semantic_expansions = {
        # Question types
        "what": ["definition", "meaning", "explanation", "description", "details"],
        "how": ["process", "method", "procedure", "way", "steps", "mechanism"],
        "when": ["time", "period", "duration", "date", "timing", "schedule"],
        "where": ["location", "place", "position", "site", "area"],
        "why": ["reason", "purpose", "cause", "rationale", "explanation"],
        "who": ["person", "individual", "entity", "party", "responsible"],
        
        # Common concepts
        "cost": ["price", "fee", "charge", "amount", "expense", "payment"],
        "limit": ["maximum", "minimum", "cap", "threshold", "boundary", "restriction"],
        "time": ["period", "duration", "timeframe", "schedule", "timing"],
        "cover": ["include", "protection", "benefit", "coverage", "encompass"],
        "require": ["need", "necessary", "condition", "prerequisite", "mandate"],
        "allow": ["permit", "enable", "authorize", "approve", "accept"],
        "provide": ["offer", "supply", "give", "deliver", "furnish"],
        "include": ["contain", "comprise", "encompass", "cover", "feature"],
        "exclude": ["omit", "except", "not include", "leave out", "bar"],
        "apply": ["relevant", "applicable", "pertain", "relate", "concern"],
        "available": ["accessible", "obtainable", "offered", "provided"],
        "eligible": ["qualified", "entitled", "suitable", "acceptable"],
        "benefit": ["advantage", "service", "feature", "coverage", "perk"],
        "condition": ["requirement", "criteria", "term", "stipulation", "clause"],
        "process": ["procedure", "method", "steps", "workflow", "system"],
        "document": ["paper", "form", "certificate", "record", "file"],
        "information": ["data", "details", "facts", "particulars", "specifics"]
    }
    
    # Add semantic variations for question words
    for word in question_words:
        if word in semantic_expansions:
            keywords.extend(semantic_expansions[word])
        else:
            # Add the word itself and common variations
            keywords.append(word)
            if word.endswith('s'):
                keywords.append(word[:-1])  # Remove plural
            elif not word.endswith('s'):
                keywords.append(word + 's')  # Add plural
    
    # Add domain-specific expansions based on document context
    doc_type = document_context.get("type", "general")
    domain_terms = document_context.get("domain_terms", [])
    
    # Include relevant domain terms
    for term in domain_terms[:10]:  # Top 10 domain-specific terms
        if any(qword in term or term in qword for qword in question_words):
            keywords.append(term)
    
    # Domain-specific keyword expansion
    if doc_type == "insurance":
        insurance_keywords = []
        for word in question_words:
            if word in ["coverage", "cover", "covered"]:
                insurance_keywords.extend(["policy", "benefit", "protection", "insurance", "plan"])
            elif word in ["cost", "price", "fee"]:
                insurance_keywords.extend(["premium", "deductible", "copay", "out-of-pocket"])
            elif word in ["claim", "claims"]:
                insurance_keywords.extend(["reimbursement", "payment", "settlement", "compensation"])
            elif word in ["exclude", "exclusion", "excluded"]:
                insurance_keywords.extend(["not covered", "limitation", "restriction", "exception"])
            elif word in ["wait", "waiting"]:
                insurance_keywords.extend(["period", "time", "duration", "delay"])
        keywords.extend(insurance_keywords)
    
    elif doc_type == "legal":
        legal_keywords = []
        for word in question_words:
            if word in ["contract", "agreement"]:
                legal_keywords.extend(["terms", "conditions", "clause", "provision"])
            elif word in ["liable", "liability"]:
                legal_keywords.extend(["responsible", "obligation", "duty", "accountability"])
            elif word in ["breach", "violation"]:
                legal_keywords.extend(["default", "non-compliance", "failure", "infringement"])
        keywords.extend(legal_keywords)
    
    elif doc_type == "medical":
        medical_keywords = []
        for word in question_words:
            if word in ["treatment", "treat"]:
                medical_keywords.extend(["therapy", "care", "procedure", "intervention"])
            elif word in ["diagnosis", "diagnose"]:
                medical_keywords.extend(["condition", "disease", "illness", "disorder"])
            elif word in ["medication", "medicine"]:
                medical_keywords.extend(["drug", "prescription", "pharmaceutical"])
        keywords.extend(medical_keywords)
    
    # Remove duplicates while preserving order
    unique_keywords = []
    seen = set()
    for keyword in keywords:
        if keyword.lower() not in seen:
            unique_keywords.append(keyword)
            seen.add(keyword.lower())
    
    # Limit to reasonable number for performance
    return unique_keywords[:15]

async def generate_hackrx_answer(question: str, clauses: List[Clause], document_context: Dict[str, Any] = None) -> str:
    """Generate intelligent, context-aware answers for any document type"""
    
    if not document_context:
        # Quick context analysis if not provided
        document_context = analyze_document_context(clauses)
    
    # Select optimal number of clauses based on question complexity and document type
    question_complexity = len(question.split())
    doc_type = document_context.get("type", "general")
    
    if question_complexity > 15 or doc_type in ["legal", "technical"]:
        relevant_clauses = clauses[:6]  # More context for complex questions
    elif question_complexity > 10 or doc_type in ["insurance", "medical"]:
        relevant_clauses = clauses[:5]  # Standard context
    else:
        relevant_clauses = clauses[:4]  # Focused context for simple questions
    
    # Prepare context with intelligent clause ranking
    clauses_context = "\n".join([
        f"Source {i+1} (Relevance: {clause.score:.2f}): {clause.content}"
        for i, clause in enumerate(relevant_clauses)
    ])
    
    # Generate domain-specific prompt
    domain_prompts = {
        "insurance": "You are an expert insurance policy analyst specializing in coverage, claims, and policy terms.",
        "legal": "You are an expert legal document analyst specializing in contracts, agreements, and legal provisions.",
        "medical": "You are an expert medical document analyst specializing in healthcare policies, treatments, and procedures.",
        "financial": "You are an expert financial document analyst specializing in investments, returns, and financial terms.",
        "hr": "You are an expert HR policy analyst specializing in employee benefits, policies, and procedures.",
        "technical": "You are an expert technical documentation analyst specializing in systems, procedures, and specifications.",
        "academic": "You are an expert academic document analyst specializing in research, studies, and academic content.",
        "business": "You are an expert business document analyst specializing in operations, strategies, and business processes.",
        "general": "You are an expert document analyst with broad knowledge across multiple domains."
    }
    
    expert_role = domain_prompts.get(doc_type, domain_prompts["general"])
    
    # Create intelligent prompt based on question type
    question_lower = question.lower()
    
    if question_lower.startswith(("what is", "what are", "define", "definition")):
        instruction_focus = "Provide a clear, precise definition with specific details from the sources."
    elif question_lower.startswith(("how much", "what is the cost", "what is the amount")):
        instruction_focus = "Provide exact amounts, costs, or numerical values with any conditions or limitations."
    elif question_lower.startswith(("how", "what is the process", "what are the steps")):
        instruction_focus = "Explain the process or procedure with specific steps and requirements."
    elif question_lower.startswith(("when", "what is the time", "what is the period")):
        instruction_focus = "Provide specific timeframes, periods, or dates with any relevant conditions."
    elif question_lower.startswith(("where", "what is the location")):
        instruction_focus = "Provide specific location or place information with relevant details."
    elif question_lower.startswith(("why", "what is the reason")):
        instruction_focus = "Explain the reasoning or purpose based on the document content."
    elif question_lower.startswith(("who", "which person", "which entity")):
        instruction_focus = "Identify the specific person, entity, or responsible party."
    elif "covered" in question_lower or "coverage" in question_lower:
        instruction_focus = "Clearly state what is covered or not covered with specific conditions."
    elif "exclude" in question_lower or "exclusion" in question_lower:
        instruction_focus = "List specific exclusions or limitations with exact conditions."
    else:
        instruction_focus = "Provide a direct, factual answer based on the document content."
    
    prompt = f"""{expert_role} Analyze the document sources and provide an accurate answer.

DOCUMENT SOURCES:
{clauses_context}

QUESTION: {question}

ANALYSIS INSTRUCTIONS:
- {instruction_focus}
- Answer in 1-2 clear, concise sentences maximum
- Use exact numbers, percentages, timeframes, and amounts from the sources
- Quote specific conditions, requirements, or limitations as written
- If information is not found in the sources, respond: "Not specified in policy"
- Be completely accurate - only state what is explicitly written
- Do not make assumptions or interpretations beyond the source content
- Include relevant context that directly answers the question

EXPERT ANSWER:"""

    try:
        # Optimized generation parameters for accuracy and relevance
        generation_config = {
            'temperature': 0.05,  # Very low for consistency, slightly higher for natural language
            'top_p': 0.95,
            'top_k': 2,
            'max_output_tokens': 120,  # Slightly more tokens for comprehensive answers
        }
        
        response = gemini_service.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        answer = response.text.strip()
        
        # Clean and validate answer
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        
        # Remove common AI response prefixes
        prefixes_to_remove = [
            "Based on the provided sources, ",
            "According to the document, ",
            "The document states that ",
            "From the sources provided, ",
            "The policy indicates that "
        ]
        
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):]
                break
        
        # Ensure answer is substantial and relevant
        if not answer or len(answer.strip()) < 5:
            return "Not specified in policy"
        
        # Check for generic non-answers
        generic_responses = [
            "the information is not available",
            "this is not mentioned",
            "no information provided",
            "cannot be determined"
        ]
        
        if any(generic in answer.lower() for generic in generic_responses):
            return "Not specified in policy"
        
        return answer
        
    except Exception as e:
        print(f"Answer generation error: {e}")
        return "Not specified in policy"

# Hackathon-specific models
class HackRXRequest(BaseModel):
    documents: str  # URL to the PDF document
    questions: List[str]  # List of questions to ask

class HackRXResponse(BaseModel):
    answers: List[str]  # List of direct answers

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process document query and return structured response"""
    try:
        # Check if Gemini service is available
        if not gemini_service:
            raise HTTPException(status_code=503, detail="Gemini service not available")
        
        # Generate cache key if Redis is available
        cached_response = None
        if redis_service:
            clause_ids = [clause.clause_id for clause in request.top_k_clauses]
            query_hash = redis_service.generate_query_hash(request.user_question, clause_ids)
            cached_response = redis_service.get_cached_response(query_hash)
        
        # Return cached response if available
        if cached_response:
            print("✅ Returning cached response")
            return cached_response
        
        # Process with Gemini if not cached
        response = gemini_service.analyze_clauses(request.user_question, request.top_k_clauses)
        
        # Cache the response if Redis is available
        if redis_service:
            redis_service.cache_response(query_hash, response)
            print("✅ Response cached for future requests")
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Document Query Assistant",
        "version": "1.0.0"
    }



@app.get("/supabase/stats")
async def get_supabase_stats():
    """Get Supabase database statistics"""
    try:
        stats = await supabase_db_service.get_performance_stats()
        return {
            "status": "success",
            "database": "Supabase",
            "database_stats": stats,
            "performance_benefits": {
                "instant_cache_hits": "Answers served in <100ms via Supabase",
                "semantic_matching": "Similar questions auto-matched",
                "document_reuse": "No reprocessing of same PDFs",
                "cloud_scalability": "Supabase handles scaling automatically",
                "real_time_sync": "Real-time updates across instances"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/supabase/optimize")
async def optimize_supabase():
    """Optimize Supabase database for better performance"""
    try:
        result = await supabase_db_service.optimize_database()
        return {
            "status": "success",
            "database": "Supabase",
            "optimization_result": result,
            "recommendations": [
                "Supabase database optimized for faster queries",
                "Old cache entries cleaned up",
                "Indexes optimized for performance",
                "Cloud scaling configured"
            ]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/supabase/recommendations/{document_hash}")
async def get_supabase_recommendations(document_hash: str):
    """Get smart clause recommendations from Supabase"""
    try:
        recommendations = await supabase_db_service.get_smart_clause_recommendations("", document_hash)
        return {
            "status": "success",
            "database": "Supabase",
            "recommendations": recommendations,
            "explanation": "These clauses are most frequently accessed and highly rated in Supabase"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/test-redis")
async def test_redis():
    """Test Redis connection endpoint"""
    if not redis_service:
        raise HTTPException(status_code=503, detail="Redis service not available")
    
    result = redis_service.test_connection()
    if result["status"] == "error":
        raise HTTPException(status_code=503, detail=result["message"])
    return result

@app.get("/cache/stats")
async def get_cache_stats():
    """Get document cache statistics"""
    if not redis_service:
        raise HTTPException(status_code=503, detail="Redis service not available")
    
    try:
        stats = redis_service.get_document_cache_stats()
        return {
            "status": "success",
            "cache_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache stats: {str(e)}")

@app.delete("/cache/clear")
async def clear_cache():
    """Clear document cache"""
    if not redis_service:
        raise HTTPException(status_code=503, detail="Redis service not available")
    
    try:
        success = redis_service.clear_document_cache()
        if success:
            return {"status": "success", "message": "Document cache cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear document cache")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(request: HackRXRequest):
    """LIGHTNING-FAST: Real-time PDF analysis with maximum speed"""
    try:
        print(f"⚡ LIGHTNING REQUEST: Processing {len(request.questions)} questions")
        
        # Check services availability
        if not gemini_service:
            raise HTTPException(status_code=503, detail="Gemini service not available")
        if not pinecone_service or not pinecone_service.index:
            raise HTTPException(status_code=503, detail="Pinecone service not available")
        
        # LIGHTNING-FAST PROCESSING: Direct approach for maximum speed
        print("🚀 Using Lightning Processor for maximum speed...")
        
        answers = await lightning_processor.process_lightning_fast(
            request.documents,
            request.questions,
            pinecone_service,
            redis_service,
            supabase_db_service
        )
        
        return HackRXResponse(answers=answers)
        
        # HIGH-PERFORMANCE DATABASE CACHE: Lightning-fast document retrieval
        import hashlib, time, uuid

        document_name: str | None = None
        clauses: List[Clause] | None = None
        url_hash = hashlib.md5(request.documents.encode()).hexdigest()[:16]

        # Check high-performance Supabase database first (fastest)
        if await supabase_db_service.is_document_cached(url_hash):
            doc_info = await supabase_db_service.get_cached_document_info(url_hash)
            cached_clauses = await supabase_db_service.get_cached_clauses(url_hash)
            
            if doc_info and cached_clauses:
                document_name = doc_info["document_name"]
                
                def _db_to_clause(c: dict) -> Clause:
                    return Clause(
                        clause_id=c["id"],
                        content=c["content"],
                        score=c.get("score", 1.0)
                    )
                
                clauses = [_db_to_clause(c) for c in cached_clauses]
                print(f"🚀 SUPABASE CACHE HIT: {document_name} with {len(clauses)} clauses")
                print(f"📊 Document intelligence: {doc_info.get('intelligence', {})}")
        
        # Fallback to Redis cache if database doesn't have it
        elif redis_service and redis_service.is_document_cached(request.documents):
            info = redis_service.get_cached_document_info(request.documents)
            cached = redis_service.get_cached_clauses(request.documents)
            if info and cached:
                document_name = info.get("document_name")
                def _cached_to_clause(c: dict) -> Clause:
                    return Clause(
                        clause_id=c.get('clause_id') or c.get('id'),
                        content=c.get('content', ''),
                        score=c.get('score', 1.0)
                    )
                clauses = [_cached_to_clause(c) for c in cached]
                print(f"✅ Redis cache hit: {document_name} with {len(clauses)} clauses")
                
                # Store in Supabase for future speed
                doc_data = {
                    "url_hash": url_hash,
                    "document_name": document_name,
                    "original_url": request.documents,
                    "domain_type": "general",
                    "file_size": info.get("file_size")
                }
                clause_dicts = [{"id": c.clause_id, "content": c.content, "score": c.score} for c in clauses]
                await supabase_db_service.store_document_with_clauses(doc_data, clause_dicts)
                print("📈 Migrated to Supabase for future speed")

        if clauses is None:
            # HIGH-PERFORMANCE PROCESSING with Database Integration
            document_name = f"doc_{url_hash}"
            print(f"🆕 HIGH-PERFORMANCE PROCESSING: New document {document_name}")
            print(f"🔗 PDF URL: {request.documents}")
            
            if clauses is None:
                print("⚡ DOWNLOADING PDF (streaming mode)...")
                async with httpx.AsyncClient(timeout=30) as client:
                    pdf_response = await client.get(request.documents)
                pdf_response.raise_for_status()

                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(pdf_response.content)
                    temp_file_path = temp_file.name

                try:
                    # Mark processing started in Supabase
                    if supabase_service and supabase_service.is_available():
                        await supabase_service.mark_processing_started(url_hash, document_name)
                    
                    # Quick document analysis for context (minimal memory usage)
                    print("🧠 QUICK ANALYSIS for context...")
                    smart_processor = SmartDocumentProcessor()
                    
                    # Extract just first few pages for analysis
                    sample_text = ""
                    try:
                        import pdfplumber
                        with pdfplumber.open(temp_file_path) as pdf:
                            # Analyze only first 3 pages for context
                            for i, page in enumerate(pdf.pages[:3]):
                                if page.extract_text():
                                    sample_text += page.extract_text() + "\n"
                    except:
                        pass
                    
                    # Quick intelligence analysis
                    sample_data = {
                        "text": sample_text,
                        "pages": [],
                        "structure": {"sections": [], "tables": [], "lists": []},
                        "metadata": {"total_pages": 1, "file_type": ".pdf"}
                    }
                    intelligence = smart_processor.analyze_document_intelligence(sample_data)
                    
                    print(f"📊 CONTEXT ANALYSIS:")
                    print(f"   Domain: {intelligence['domain']['primary_domain']} (confidence: {intelligence['domain']['confidence']:.2f})")
                    print(f"   Complexity: {intelligence['complexity']['level']}")
                    
                    # Clear sample data from memory
                    del sample_text, sample_data
                    import gc
                    gc.collect()
                    
                    # Use STREAMING processor for memory-efficient processing
                    streaming_processor = StreamingDocumentProcessor(chunk_size=3, max_workers=2)  # Small chunks for memory efficiency
                    
                    print("🌊 STREAMING PROCESSING: Processing document in chunks...")
                    
                    # Process document with streaming (memory efficient)
                    result = await streaming_processor.process_document_streaming(
                        temp_file_path, 
                        document_name, 
                        intelligence['domain'],  # Pass domain context
                        pinecone_service,
                        supabase_service
                    )
                    
                    if result["success"]:
                        print(f"✅ STREAMING SUCCESS: {result['total_clauses']} clauses processed")
                        
                        # Store in SUPABASE for lightning-fast future access
                        doc_data = {
                            "url_hash": url_hash,
                            "document_name": document_name,
                            "original_url": request.documents,
                            "domain_type": intelligence['domain']['primary_domain'],
                            "complexity_level": intelligence['complexity']['level'],
                            "file_size": len(pdf_response.content),
                            "intelligence": {
                                "domain": intelligence['domain'],
                                "complexity": intelligence['complexity'],
                                "key_terms": intelligence.get('key_terms', [])[:20]
                            }
                        }
                        
                        # Store ONLY document metadata in Supabase (not full content)
                        # Clauses are already in Pinecone, we just store references and metadata
                        clause_metadata = []
                        for i, clause_id in enumerate(result['clause_ids']):
                            # Store only metadata, not full content
                            clause_meta = {
                                "id": clause_id,
                                "content": f"[Clause stored in Pinecone: {clause_id}]",  # Reference only
                                "score": 1.0,
                                "section": i + 1,
                                "section_title": f"Section {i + 1}",
                                "metadata": {
                                    "document_type": intelligence['domain']['primary_domain'],
                                    "processing_method": "streaming",
                                    "word_count": 0,  # Will be updated when needed
                                    "char_count": 0,   # Will be updated when needed
                                    "stored_in": "pinecone"  # Indicates where full content is
                                }
                            }
                            clause_metadata.append(clause_meta)
                        
                        # Store ONLY metadata in Supabase (efficient storage)
                        await supabase_db_service.store_document_with_clauses(doc_data, clause_metadata)
                        print(f"💾 Stored metadata for {len(clause_metadata)} clauses in Supabase")
                        
                        # Create clause objects for immediate use
                        clauses = []
                        for clause_dict in processed_clauses[:50]:  # Limit for immediate use
                            clause = Clause(
                                clause_id=clause_dict["id"],
                                content=clause_dict["content"],
                                score=clause_dict["score"]
                            )
                            clauses.append(clause)
                        
                        # Also cache in Redis for compatibility
                        if redis_service:
                            cache_data = [{"id": c.clause_id, "content": c.content, "score": c.score} for c in clauses]
                            redis_service.cache_document(request.documents, document_name, cache_data,
                                                         file_size=len(pdf_response.content))
                            redis_service.mark_embeddings_uploaded(request.documents)
                        
                        print("🚀 SUPABASE STORAGE COMPLETE")
                    else:
                        raise Exception(f"Streaming processing failed: {result.get('error', 'Unknown error')}")
                        
                    # Clean up processor memory
                    streaming_processor.cleanup_memory()
                    del streaming_processor
                    gc.collect()
                    
                finally:
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)

        # At this point, clauses for this PDF are guaranteed to be embedded and searchable in Pinecone.
        # Proceed to answer the user’s questions.
            


        
        # Process questions in parallel for maximum speed
        import asyncio
        
        async def process_single_question(question: str, question_index: int) -> str:
            """HIGH-PERFORMANCE: Process question with intelligent caching and database optimization"""
            
            try:
                start_time = time.time()
                
                # STEP 1: Check SUPABASE CACHE for instant response
                cached_answer = await supabase_db_service.get_cached_answer(question, url_hash)
                if cached_answer:
                    print(f"⚡ SUPABASE CACHE HIT: Q{question_index} - {cached_answer['hit_count']} previous hits")
                    return cached_answer["answer"]
                
                # STEP 2: Check for similar cached answers (semantic matching)
                similar_answers = await supabase_db_service.find_similar_cached_answers(question, url_hash, 0.85)
                if similar_answers:
                    best_match = similar_answers[0]
                    print(f"🎯 SEMANTIC CACHE HIT: Q{question_index} - {best_match['similarity']:.2f} similarity")
                    return best_match["answer"]
                
                # STEP 3: Analyze document context for intelligent processing
                document_context = analyze_document_context(clauses)
                print(f"📊 Document analysis: Type={document_context['type']}, Terms={len(document_context['domain_terms'])}")
                
                # STEP 4: Advanced question preprocessing with context awareness
                processed_question = preprocess_question(question, document_context)
                
                print(f"🔍 SMART PROCESSING: Q{question_index} - {processed_question[:50]}...")
                print(f"🎯 Original: {question[:50]}...")
                
                # STEP 3: Generate intelligent search strategies
                search_keywords = generate_search_keywords(processed_question, document_context)
                print(f"🔑 Generated {len(search_keywords)} smart keywords")
                
                # STEP 4: Multi-strategy semantic search
                genai.configure(api_key=config.GEMINI_API_KEY)
                
                all_relevant_clauses = []
                seen_content = set()
                search_scores = {}
                
                # Search with intelligent keyword prioritization
                for i, keyword in enumerate(search_keywords[:5]):  # Top 5 keywords for comprehensive search
                    try:
                        result = genai.embed_content(
                            model="models/embedding-001",
                            content=keyword,
                            task_type="retrieval_query"
                        )
                        query_embedding = result['embedding']
                        
                        # Dynamic top_k based on document context and question complexity
                        question_complexity = len(processed_question.split())
                        base_top_k = 15
                        
                        # Adjust based on document type
                        if document_context['type'] in ['insurance', 'legal']:
                            base_top_k = 20  # More comprehensive for complex domains
                        elif document_context['type'] in ['technical', 'medical']:
                            base_top_k = 18  # Moderate for technical content
                        
                        # Adjust based on question complexity
                        if question_complexity > 10:
                            base_top_k += 5
                        
                        top_k = min(base_top_k, 25)  # Cap at 25 for performance
                        
                        search_results = pinecone_service.index.query(
                            vector=query_embedding,
                            top_k=top_k,
                            include_metadata=True,
                            filter={"document": document_name}
                        )
                        
                        # Collect unique, high-quality clauses with weighted scoring
                        keyword_weight = 1.0 - (i * 0.1)  # First keywords get higher weight
                        
                        for match in search_results.matches:
                            content = match.metadata.get('content', '')
                            if content and content not in seen_content:
                                # Dynamic score threshold based on search iteration
                                min_score = 0.65 - (i * 0.05)  # Lower threshold for later searches
                                
                                if match.score > min_score:
                                    weighted_score = match.score * keyword_weight
                                    
                                    clause = Clause(
                                        clause_id=match.id,
                                        content=content,
                                        score=weighted_score
                                    )
                                    all_relevant_clauses.append(clause)
                                    seen_content.add(content)
                                    search_scores[match.id] = {
                                        'original_score': match.score,
                                        'weighted_score': weighted_score,
                                        'keyword': keyword
                                    }
                    
                    except Exception as search_error:
                        print(f"Search error for keyword '{keyword}': {search_error}")
                        continue
                
                # STEP 5: Fallback search if no results
                if not all_relevant_clauses:
                    print("🔄 No results from keyword search, trying fallback...")
                    try:
                        # Try with original question
                        result = genai.embed_content(
                            model="models/embedding-001",
                            content=question,  # Use original question
                            task_type="retrieval_query"
                        )
                        query_embedding = result['embedding']
                        
                        search_results = pinecone_service.index.query(
                            vector=query_embedding,
                            top_k=15,
                            include_metadata=True,
                            filter={"document": document_name}
                        )
                        
                        all_relevant_clauses = [
                            Clause(
                                clause_id=match.id,
                                content=match.metadata.get('content', ''),
                                score=match.score
                            )
                            for match in search_results.matches
                            if match.score > 0.5  # Lower threshold for fallback
                        ]
                    except Exception as fallback_error:
                        print(f"Fallback search error: {fallback_error}")
                        return "Error processing question - please try again"
                
                if not all_relevant_clauses:
                    return "Not specified in policy"
                
                # STEP 6: Intelligent clause ranking and selection
                all_relevant_clauses.sort(key=lambda x: x.score, reverse=True)
                
                # Select top clauses with diversity (avoid too similar content)
                selected_clauses = []
                for clause in all_relevant_clauses:
                    if len(selected_clauses) >= 8:  # Max 8 clauses
                        break
                    
                    # Check content diversity
                    is_diverse = True
                    for selected in selected_clauses:
                        # Simple similarity check
                        common_words = set(clause.content.lower().split()) & set(selected.content.lower().split())
                        if len(common_words) > len(clause.content.split()) * 0.7:  # 70% overlap
                            is_diverse = False
                            break
                    
                    if is_diverse:
                        selected_clauses.append(clause)
                
                print(f"✅ Selected {len(selected_clauses)} diverse clauses for analysis")
                
                # STEP 7: Generate contextually aware answer
                answer = await generate_hackrx_answer(processed_question, selected_clauses, document_context)
                
                # STEP 8: Intelligent answer validation and caching
                processing_time = time.time() - start_time
                
                if validate_answer_matches_question(processed_question, answer, document_context):
                    # Cache successful answer in Supabase for future instant responses
                    clause_ids = [clause.clause_id for clause in selected_clauses]
                    await supabase_db_service.cache_answer(
                        question, url_hash, answer, clause_ids, 
                        processing_time, confidence_score=0.9
                    )
                    print(f"💾 CACHED in Supabase: Q{question_index} ({processing_time:.2f}s)")
                    return answer
                else:
                    # Try with different clause combination
                    if len(all_relevant_clauses) > len(selected_clauses):
                        alternative_clauses = all_relevant_clauses[2:min(10, len(all_relevant_clauses))]
                        alternative_answer = await generate_hackrx_answer(processed_question, alternative_clauses, document_context)
                        if validate_answer_matches_question(processed_question, alternative_answer, document_context):
                            # Cache alternative answer
                            alt_clause_ids = [clause.clause_id for clause in alternative_clauses]
                            await supabase_db_service.cache_answer(
                                question, url_hash, alternative_answer, alt_clause_ids,
                                processing_time, confidence_score=0.8
                            )
                            return alternative_answer
                    
                    # Cache "not found" to avoid reprocessing
                    await supabase_db_service.cache_answer(
                        question, url_hash, "Not specified in policy", [],
                        processing_time, confidence_score=0.5
                    )
                    return "Not specified in policy"
                
            except Exception as e:
                print(f"Error processing question {question_index}: {e}")
                return "Error processing question - please try again"

        def validate_answer_matches_question(question: str, answer: str, document_context: Dict[str, Any] = None) -> bool:
            """INTELLIGENT: Context-aware validation that answer matches question intent"""
            question_lower = question.lower()
            answer_lower = answer.lower()
            
            if not document_context:
                document_context = {"type": "general", "domain_terms": []}
            
            # Extract key concepts from question
            import re
            question_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question_lower))
            answer_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', answer_lower))
            
            # Check for generic error responses first
            generic_errors = [
                "the provided policy excerpt states that if the premium",
                "error processing",
                "not found in the document",
                "unable to determine"
            ]
            
            for error_pattern in generic_errors:
                if error_pattern in answer_lower:
                    return False
            
            # If answer is too short or generic, it's likely not accurate
            if len(answer.strip()) < 10 or answer_lower in ["yes", "no", "not applicable", "n/a"]:
                return False
            
            # Context-aware semantic validation
            doc_type = document_context.get("type", "general")
            domain_terms = set(term.lower() for term in document_context.get("domain_terms", []))
            
            # Calculate semantic overlap
            common_words = question_words & answer_words
            semantic_overlap = len(common_words) / max(len(question_words), 1)
            
            # Domain-specific validation rules
            validation_rules = {
                "insurance": {
                    "coverage": ["cover", "benefit", "protection", "policy", "include"],
                    "cost": ["premium", "fee", "charge", "amount", "price", "payment"],
                    "claim": ["reimbursement", "payment", "settlement", "process"],
                    "exclusion": ["exclude", "not covered", "limitation", "restriction"],
                    "waiting": ["period", "time", "duration", "wait"],
                    "limit": ["maximum", "minimum", "cap", "threshold"]
                },
                "legal": {
                    "contract": ["agreement", "terms", "conditions", "clause"],
                    "liability": ["responsible", "obligation", "duty", "liable"],
                    "breach": ["violation", "default", "non-compliance"],
                    "jurisdiction": ["court", "authority", "legal"]
                },
                "medical": {
                    "treatment": ["therapy", "care", "procedure", "medical"],
                    "diagnosis": ["condition", "disease", "illness"],
                    "medication": ["drug", "medicine", "prescription"],
                    "patient": ["individual", "person", "subject"]
                },
                "financial": {
                    "investment": ["asset", "portfolio", "fund", "security"],
                    "return": ["profit", "gain", "yield", "income"],
                    "risk": ["volatility", "exposure", "uncertainty"]
                }
            }
            
            # Apply domain-specific validation
            if doc_type in validation_rules:
                domain_rules = validation_rules[doc_type]
                
                for concept, related_terms in domain_rules.items():
                    if concept in question_lower:
                        # Check if answer contains related terms
                        has_related_term = any(term in answer_lower for term in related_terms)
                        if not has_related_term:
                            # Check if any domain terms are present
                            has_domain_term = any(term in answer_lower for term in domain_terms)
                            if not has_domain_term and semantic_overlap < 0.2:
                                print(f"❌ Validation failed: '{concept}' question lacks related terms")
                                return False
            
            # Question type validation
            question_types = {
                "what": ["definition", "description", "explanation", "meaning"],
                "how": ["process", "method", "procedure", "way", "steps"],
                "when": ["time", "date", "period", "schedule", "timing"],
                "where": ["location", "place", "address", "site"],
                "why": ["reason", "because", "purpose", "cause"],
                "who": ["person", "individual", "entity", "responsible"],
                "how much": ["amount", "cost", "price", "fee", "charge"],
                "how many": ["number", "quantity", "count"]
            }
            
            for q_type, expected_terms in question_types.items():
                if question_lower.startswith(q_type):
                    has_expected = any(term in answer_lower for term in expected_terms)
                    if not has_expected and semantic_overlap < 0.15:
                        # Allow if answer contains specific numbers, dates, or domain terms
                        has_specifics = (
                            re.search(r'\d+', answer) or  # Contains numbers
                            re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', answer_lower) or  # Contains months
                            any(term in answer_lower for term in domain_terms)  # Contains domain terms
                        )
                        if not has_specifics:
                            print(f"❌ Validation failed: '{q_type}' question type mismatch")
                            return False
            
            # Minimum semantic overlap requirement
            if semantic_overlap < 0.1 and len(common_words) < 2:
                # Check if answer contains domain-specific terms
                domain_overlap = len(domain_terms & answer_words)
                if domain_overlap == 0:
                    print(f"❌ Validation failed: Insufficient semantic overlap ({semantic_overlap:.2f})")
                    return False
            
            # Check for contradictory responses
            contradictions = [
                ("yes" in answer_lower and "no" in answer_lower),
                ("covered" in answer_lower and "not covered" in answer_lower),
                ("included" in answer_lower and "excluded" in answer_lower),
                ("allowed" in answer_lower and "not allowed" in answer_lower)
            ]
            
            if any(contradictions):
                print("❌ Validation failed: Contradictory statements in answer")
                return False
            
            print(f"✅ Validation passed: Overlap={semantic_overlap:.2f}, Common={len(common_words)}")
            return True

        # Process all questions concurrently with increased parallelism for speed
        semaphore = asyncio.Semaphore(5)  # Increased for faster processing
        
        async def process_with_semaphore(question: str, index: int) -> str:
            async with semaphore:
                return await process_single_question(question, index + 1)
        
        # Create tasks for all questions
        tasks = [
            process_with_semaphore(question, i) 
            for i, question in enumerate(request.questions)
        ]
        
        # Execute all tasks concurrently with timeout for reliability
        try:
            answers = await asyncio.wait_for(
                asyncio.gather(*tasks), 
                timeout=30.0  # 30 second timeout for all questions
            )
        except asyncio.TimeoutError:
            # Fallback: process questions sequentially if timeout
            answers = []
            for i, question in enumerate(request.questions):
                try:
                    answer = await asyncio.wait_for(
                        process_single_question(question, i + 1),
                        timeout=5.0  # 5 second timeout per question
                    )
                    answers.append(answer)
                except asyncio.TimeoutError:
                    answers.append("Response timeout - please try again")
        
        return HackRXResponse(answers=answers)
                
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    except Exception as e:
        print(f"❌ HackRX processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
