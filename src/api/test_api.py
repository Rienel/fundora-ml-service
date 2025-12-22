import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n1. Testing Health Endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_recommendations():
    """Test recommendations endpoint"""
    print("\n2. Testing Recommendations Endpoint...")
    
    payload = {
        "user_id": 1,
        "n_recommendations": 5,
        "exclude_viewed": True
    }
    
    response = requests.post(
        f"{BASE_URL}/api/recommendations",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nRecommendations for User {data['user_id']}:")
        for i, rec in enumerate(data['recommendations'], 1):
            print(f"{i}. {rec['company_name']} (Score: {rec['score']:.2f})")
    else:
        print(f"Error: {response.json()}")
    
    return response.status_code == 200

def test_popular():
    """Test popular startups endpoint"""
    print("\n3. Testing Popular Startups Endpoint...")
    
    response = requests.get(f"{BASE_URL}/api/popular?n=5")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nTop {data['count']} Popular Startups:")
        for i, startup in enumerate(data['popular_startups'], 1):
            print(f"{i}. {startup['company_name']} (Score: {startup['score']:.2f})")
    else:
        print(f"Error: {response.json()}")
    
    return response.status_code == 200

def test_feedback():
    """Test feedback endpoint"""
    print("\n4. Testing Feedback Endpoint...")
    
    payload = {
        "user_id": 1,
        "startup_id": 2,
        "action": "watchlist"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/feedback",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

if __name__ == "__main__":
    print("="*60)
    print("TESTING FUNDORA ML API")
    print("="*60)
    
    tests = [
        ("Health Check", test_health),
        ("Recommendations", test_recommendations),
        ("Popular Startups", test_popular),
        ("Feedback", test_feedback)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"Error: {e}")
            results.append((test_name, "ERROR"))
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    for test_name, result in results:
        print(f"{test_name}: {result}")
    print("="*60)