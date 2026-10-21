"""Unit tests for prompt template management and dynamic persona name injection."""

import glob

import yaml

from agentverse.config.settings import Settings
from agentverse.services.langfuse.prompts import PromptManager


def test_all_yaml_templates_have_dynamic_assistant_name():
    """Verify that no prompt templates in configs/prompts hardcode 'Mora' in their template."""
    prompt_files = glob.glob("configs/prompts/*.yaml")
    assert len(prompt_files) >= 8

    for file_path in prompt_files:
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        template = data.get("template", "")
        # Template must not hardcode 'Your name is Mora.'
        assert "Your name is Mora" not in template, f"File {file_path} hardcodes Mora!"
        # Template must use dynamic placeholder
        assert "{assistant_name}" in template, (
            f"File {file_path} missing {{assistant_name}} placeholder!"
        )
        # Variables list must declare assistant_name
        variables = data.get("variables", [])
        assert "assistant_name" in variables, (
            f"File {file_path} missing assistant_name in variables!"
        )


def test_prompt_manager_renders_default_persona_name():
    """Verify PromptManager injects default agent_persona_name ('Mora') from settings."""
    settings = Settings(agent_persona_name="Mora")
    manager = PromptManager(settings)

    rendered = manager.render(
        "historical_chat",
        context="Sample context",
        history="User: Hi",
        question="Who was Alexander?",
    )
    assert "Your name is Mora." in rendered
    assert "Context:\nSample context" in rendered
    assert "User: Hi" in rendered


def test_prompt_manager_renders_custom_settings_persona_name():
    """Verify changing agent_persona_name dynamically changes the prompt without editing YAML."""
    settings = Settings(agent_persona_name="Hypatia")
    manager = PromptManager(settings)

    rendered = manager.render(
        "historical_chat",
        context="Sample context",
        history="No prior conversation.",
        question="Tell me about astronomy in Alexandria.",
    )
    assert "Your name is Hypatia." in rendered
    assert "Your name is Mora." not in rendered


def test_prompt_manager_explicit_persona_override():
    """Verify caller can override the assistant name at call time."""
    settings = Settings(agent_persona_name="Mora")
    manager = PromptManager(settings)

    rendered = manager.render(
        "rag_qa",
        assistant_name="Athena",
        context="Greek history",
        history="",
        question="What is the Parthenon?",
    )
    assert "Your name is Athena." in rendered
    assert "Mora" not in rendered


def test_prompt_manager_render_vision_and_voice_prompts():
    """Verify image, video, and voice prompts dynamically render persona name."""
    settings = Settings(agent_persona_name="Cleopatra")
    manager = PromptManager(settings)

    img_rendered = manager.render("image_generation", prompt="Ancient Alexandria port")
    assert "Your name is Cleopatra." in img_rendered
    assert "Ancient Alexandria port" in img_rendered

    video_rendered = manager.render("video_generation", prompt="Pharos lighthouse at sunset")
    assert "Your name is Cleopatra." in video_rendered
    assert "Pharos lighthouse at sunset" in video_rendered

    voice_rendered = manager.render(
        "voice_sts", context="Lighthouse history", question="How tall was it?"
    )
    assert "Your name is Cleopatra." in voice_rendered
    assert "How tall was it?" in voice_rendered


def test_prompt_manager_fallback_template_uses_dynamic_persona():
    """Verify fallback template dynamically injects persona name."""
    settings = Settings(agent_persona_name="Thoth")
    manager = PromptManager(settings)

    rendered = manager.render(
        "nonexistent_template_xyz",
        context="Ancient wisdom",
        history="",
        question="What is writing?",
    )
    assert "Your name is Thoth." in rendered
    assert "Mora" not in rendered


def test_all_yaml_templates_declare_model_name():
    """Verify that all prompt templates declare model_name as an available variable."""
    prompt_files = glob.glob("configs/prompts/*.yaml")
    for file_path in prompt_files:
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        variables = data.get("variables", [])
        assert "model_name" in variables, f"File {file_path} missing model_name in variables!"


def test_prompt_manager_injects_model_name_variable():
    """Verify PromptManager automatically injects model_name from settings or overrides."""
    settings = Settings(agent_persona_name="Mora", llm_model_name="gemini/gemini-1.5-pro")
    manager = PromptManager(settings)

    manager._cache["test_model_prompt:v1.0"] = (
        "Assistant {assistant_name} running on model {model_name}."
    )
    rendered = manager.render("test_model_prompt")
    assert rendered == "Assistant Mora running on model gemini/gemini-1.5-pro."

    # With explicit override
    override_rendered = manager.render(
        "test_model_prompt", assistant_name="Athena", model_name="gpt-4o"
    )
    assert override_rendered == "Assistant Athena running on model gpt-4o."
