#!/usr/bin/env python3
"""
Test environment variable loading
"""

import os
from dotenv import load_dotenv

print("🔍 Testing environment variable loading...")

# Load .env file
load_dotenv()

print(f"SUPABASE_URL: {os.getenv('SUPABASE_URL')}")
print(f"SUPABASE_ANON_KEY: {os.getenv('SUPABASE_ANON_KEY')[:20]}..." if os.getenv('SUPABASE_ANON_KEY') else "SUPABASE_ANON_KEY: None")
print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')[:20]}..." if os.getenv('GEMINI_API_KEY') else "GEMINI_API_KEY: None")

# Test Supabase connection
try:
    from supabase import create_client
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if supabase_url and supabase_key:
        print("✅ Supabase credentials found")
        client = create_client(supabase_url, supabase_key)
        print("✅ Supabase client created successfully")
        
        # Test connection
        result = client.table("documents").select("count", count="exact").execute()
        print(f"✅ Supabase connection successful! Documents: {result.count or 0}")
        
    else:
        print("❌ Supabase credentials missing")
        
except Exception as e:
    print(f"❌ Supabase connection failed: {e}")