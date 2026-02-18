from parse_then_place.semantic_parser.dataset.text_processor import TextProcessor


def test_text_processor_preprocess_normalizes():
    processor = TextProcessor()
    text = "Hello   WORLD #Test “Quotes” "
    processed, value_map = processor.preprocess(text)
    assert processed == 'hello world test "quotes"'
    assert value_map is None


def test_text_processor_extract_explicit_values():
    processor = TextProcessor(replace_value=True)
    processed, value_map = processor.preprocess('Click "Sign In" and "Help".')
    assert processed == 'click "value_0" and "value_1".'
    assert value_map["value_0"] == "sign in"
    assert value_map["value_1"] == "help"


def test_text_processor_ignores_single_quote_with_punctuation():
    processor = TextProcessor(replace_value=True)
    processed, value_map = processor.preprocess("Tap 'alpha, beta' now")
    assert processed == "tap 'alpha, beta' now"
    assert value_map == {}
