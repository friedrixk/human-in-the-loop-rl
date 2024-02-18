# AUTO GENERATED FILE - DO NOT EDIT

from dash.development.base_component import Component, _explicitize_args


class MyDashComponent(Component):
    """A MyDashComponent component.
This component provides a text input field with highlight functionality.

Keyword arguments:

- id (string; optional):
    The ID used to identify this component in Dash callbacks.

- highlight (list | string; optional):
    The words to be highlighted.

- label (string; optional):
    A label that will be printed when this component is rendered.

- value (string; optional):
    The value displayed in the input."""
    _children_props = []
    _base_nodes = ['children']
    _namespace = 'my_dash_component'
    _type = 'MyDashComponent'
    @_explicitize_args
    def __init__(self, id=Component.UNDEFINED, label=Component.UNDEFINED, value=Component.UNDEFINED, highlight=Component.UNDEFINED, **kwargs):
        self._prop_names = ['id', 'highlight', 'label', 'value']
        self._valid_wildcard_attributes =            []
        self.available_properties = ['id', 'highlight', 'label', 'value']
        self.available_wildcard_properties =            []
        _explicit_args = kwargs.pop('_explicit_args')
        _locals = locals()
        _locals.update(kwargs)  # For wildcard attrs and excess named props
        args = {k: _locals[k] for k in _explicit_args}

        super(MyDashComponent, self).__init__(**args)
