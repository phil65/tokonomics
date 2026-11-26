"""Names other released packages import from tokonomics.model_names."""


def test_anthropic_max_names_stay_importable():
    """Agentpool <= 2.14.0 imports AnthropicMaxModelName (``AnthropicMaxModelName | str``)."""
    from tokonomics.model_names import AnthropicMaxModelId, AnthropicMaxModelName

    assert AnthropicMaxModelName | str == str | str
    assert AnthropicMaxModelId is str
