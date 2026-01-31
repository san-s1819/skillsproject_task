"""
Quick Setup Verification Script
Checks that all components are properly installed and configured
"""

import sys
import os


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ required")
        return False
    
    print("✅ Python version OK")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'chromadb',
        'sklearn',  # scikit-learn
        'requests'
    ]
    
    print("\nChecking dependencies...")
    all_ok = True
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_api_key():
    """Check if Gemini API key is set"""
    print("\nChecking API key...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        print("\nTo set your API key:")
        print("  Windows: set GEMINI_API_KEY=your_key_here")
        print("  Linux/Mac: export GEMINI_API_KEY=your_key_here")
        print("\nGet a free key at: https://aistudio.google.com/app/apikey")
        return False
    
    print(f"✅ GEMINI_API_KEY set (length: {len(api_key)})")
    return True


def check_data_directory():
    """Check if recall documents exist"""
    print("\nChecking data directory...")
    
    data_dir = "data/recalls"
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return False
    
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if len(txt_files) < 15:
        print(f"❌ Expected 15 recall documents, found {len(txt_files)}")
        return False
    
    print(f"✅ Found {len(txt_files)} recall documents")
    return True


def check_utils_modules():
    """Check if utils modules exist"""
    print("\nChecking utils modules...")
    
    modules = ['vector_store', 'llm_client', 'agent']
    all_ok = True
    
    for module in modules:
        filepath = f"utils/{module}.py"
        if os.path.exists(filepath):
            print(f"✅ {module}.py")
        else:
            print(f"❌ {module}.py - NOT FOUND")
            all_ok = False
    
    return all_ok


def main():
    """Run all checks"""
    print("=" * 60)
    print("AUTOMOTIVE RECALL AGENT - SETUP VERIFICATION")
    print("=" * 60 + "\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("API Key", check_api_key),
        ("Data Directory", check_data_directory),
        ("Utils Modules", check_utils_modules)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error during {name} check: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! You're ready to run the agent.")
        print("\nNext steps:")
        print("  1. Run the agent: python main.py")
        print("  2. Run evaluation: python evaluate.py")
    else:
        print(f"\n⚠️  {total - passed} check(s) failed. Please fix the issues above.")
        
        if not results[1][1]:  # Dependencies failed
            print("\nTo install dependencies:")
            print("  pip install -r requirements.txt")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
