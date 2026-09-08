#[test]
fn test_invalid_model_select_usage_fails_to_compile() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/*.rs");
}
