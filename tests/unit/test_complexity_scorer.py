from core.controller.complexity_scorer import classify_request

def test_reflex_classification():
    result = classify_request("what time is it")
    assert result["class"] == "Reflex"
    assert result["route"] == "direct"
    assert result["skip_planner"] is True
    assert 0.0 <= result["complexity"] <= 0.2

def test_chat_classification():
    result = classify_request("how to use a for loop in python")
    assert result["class"] == "Chat"
    assert result["route"] == "mid-tier"

def test_agentic_classification():
    result = classify_request("create a new file called test.py")
    assert result["class"] == "Agentic"
    assert result["route"] == "planner"
    assert result["needs_tools"] is True
    assert result["skip_planner"] is False

def test_deep_reasoning_classification():
    result = classify_request("debug this complex architecture problem")
    assert result["class"] == "Deep_Reasoning"
    assert result["route"] == "premium"
    assert result["needs_reasoning"] is True
    assert result["skip_planner"] is False

def test_new_keys_present():
    result = classify_request("hello")
    assert "estimated_tokens" in result
    assert "needs_reasoning" in result
    assert "needs_tools" in result
    assert "needs_vision" in result
    assert "context_weight" in result

def test_complexity_modifiers():
    short_chat = classify_request("tell me a joke")
    long_chat_text = "tell me a joke " * 100
    long_chat = classify_request(long_chat_text)
    assert long_chat["complexity"] > short_chat["complexity"]

def test_multi_part_detection():
    single = classify_request("do this")
    multi = classify_request("First do this, then do that, and also do the other thing")
    assert multi["complexity"] > single["complexity"]

def test_code_detection():
    text_without_code = "how do i sort an array"
    text_with_code = "how do i sort an array like `[1, 2, 3]`"
    no_code_res = classify_request(text_without_code)
    code_res = classify_request(text_with_code)
    assert code_res["complexity"] > no_code_res["complexity"]

def test_vision_detection():
    res = classify_request("can you describe this image.png")
    assert res["needs_vision"] is True

def test_complexity_capping():
    # Construct a very complex string
    text = "debug architecture " * 200 + " if when unless " * 50 + " api async await docker " * 50 + " ```code``` then also"
    res = classify_request(text)
    assert res["complexity"] <= 1.0

def test_type_correctness():
    res = classify_request("test input")
    assert isinstance(res["class"], str)
    assert isinstance(res["complexity"], float)
    assert isinstance(res["route"], str)
    assert isinstance(res["skip_planner"], bool)
    assert isinstance(res["estimated_tokens"], int)
    assert isinstance(res["needs_reasoning"], bool)
    assert isinstance(res["needs_tools"], bool)
    assert isinstance(res["needs_vision"], bool)
    assert isinstance(res["context_weight"], float)
