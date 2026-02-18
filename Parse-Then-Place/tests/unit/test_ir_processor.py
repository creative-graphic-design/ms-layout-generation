from parse_then_place.semantic_parser.dataset.ir_processor import IRProcessor


def test_ir_processor_preprocess_removes_values_and_renames():
    processor = IRProcessor()
    lf = "[el:text [attr:value 'Hello']]"
    processed = processor.preprocess(lf, None)
    assert processed.strip() == "[ element : text ]"


def test_ir_processor_replace_explicit_values():
    processor = IRProcessor(remove_value=False, replace_value=True)
    lf = "[ prop:value 'sign in' ] [ group_prop:names 'help, cancel' ]"
    value_map = {"sign in": "value_0", "help": "value_1"}
    replaced = processor.replace_explicit_values(lf, value_map)
    assert "[ prop:value 'value_0' ]" in replaced
    assert "[ group_prop:names 'value_1, cancel' ]" in replaced


def test_ir_processor_preprocess_executes_value_replacement_branch():
    processor = IRProcessor(remove_value=False, replace_value=True)
    lf = "[el:text [attr:value 'sign in']]"
    processed = processor.preprocess(lf, {"value_0": "sign in"})
    assert "prop" in processed


def test_ir_processor_replace_explicit_values_strips_unknown_values():
    processor = IRProcessor(remove_value=False, replace_value=True)
    lf = "[ prop:value 'sign in, now' ]"
    replaced = processor.replace_explicit_values(lf, {})
    assert "[ prop:value 'sign in, now' ]" in replaced


def test_ir_processor_postprocess_recovers_labels():
    processor = IRProcessor()
    lf = "[ region : info [ element : text ] ]"
    recovered = processor.postprocess(lf, recover_labels=True)
    assert recovered == "[region:SingleInfo[el:text]]"


def test_ir_processor_postprocess_remove_attrs():
    processor = IRProcessor()
    lf = "[ element : text [ prop : value 'hello' ] [ prop : size 'large' ] ]"
    recovered = processor.postprocess(lf, remove_attrs=True)
    assert "prop" not in recovered


def test_ir_processor_postprocess_recovers_values():
    processor = IRProcessor()
    lf = "[ region : info [ element : text [ prop : value 'value_0' ] ] ] &"
    recovered = processor.postprocess(
        lf,
        recover_labels=True,
        recover_values=True,
        value_map={"value_0": "Sign In"},
    )
    assert "Sign In" in recovered
    assert " and " in recovered
