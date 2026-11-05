import pytest


def test_factory_is_inert_and_tools_only_without_explicit_opt_in():
    from neral_control import create_shared_controller,ControllerRunResult
    controller=create_shared_controller()
    assert controller.provider is None
    result=controller.run('2+2',permissions=frozenset({'math'}))
    assert isinstance(result,ControllerRunResult) and result['model_calls']==0 and result['text']=='4'


def test_factory_requires_explicit_model_opt_in():
    from neral_control import create_shared_controller
    provider=object()
    with pytest.raises(ValueError):create_shared_controller(provider=provider)
    controller=create_shared_controller(provider=provider,allow_model_inference=True)
    assert controller.provider is provider


def test_factory_does_not_replace_explicit_registry():
    from neral_control import create_shared_controller
    from neral_control.registry import ToolRegistry
    registry=ToolRegistry()
    assert create_shared_controller(registry=registry).registry is registry
