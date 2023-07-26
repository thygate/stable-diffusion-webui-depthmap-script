import gradio as gr

class GradioComponentBundle:
    """Allows easier transportation of massive ammount of named gradio inputs.
    Allows adding visibility rules quicker."""
    def __init__(self):
        self.internal = {}
        self.internal_ignored = {}

    def _raw_assignment(self, key, value, ignored=False):
        assert key not in self.internal, f"Already bundled component with name {key}."
        assert key not in self.internal_ignored, f"Already bundled component with name {key}."
        if not ignored:
            self.internal[key] = value
        else:
            self.internal_ignored[key] = value

    def _append_el(self, thing, ignored=False):
        if isinstance(thing, tuple) and len(thing) == 2 and isinstance(thing[1], gr.blocks.Block):
            name = thing[0] if isinstance(thing[0], str) else thing[0].name.lower()  # .name is for Enums
            if hasattr(thing[0], 'df') and thing[0].df is not None:
                thing[1].value = thing[0].df
            self._raw_assignment(name, thing[1], ignored)
        elif isinstance(thing, gr.components.Component) and thing.elem_id is not None:
            self._raw_assignment(thing.elem_id, thing, ignored)
        else:
            raise Exception(f"This object can not be bundled, {str(thing)}")

    def __iadd__(self, els):
        """Add an input element that will be packed into a bundle."""
        self._append_el(els, ignored=False)
        return self

    def __isub__(self, els):
        """Add an element that will not be packed into a bundle, but will be accessible."""
        self._append_el(els, ignored=True)
        return self

    def __ior__(self, thing):
        """Add an extra bundle into your bundle, so you could have more bundeled items in your bundle."""
        assert isinstance(thing, GradioComponentBundle), "Use += or -= for bundling elements"
        for key in list(thing.internal.keys()):
            self._raw_assignment(key, thing[key], False)
        for key in list(thing.internal_ignored.keys()):
            self._raw_assignment(key, thing[key], True)
        return self

    def __getitem__(self, key):
        """Return the gradio component elem_id"""
        if hasattr(key, 'name'):
            key = key.name.lower()  # for enum elements
        if key in self.internal_ignored:
            return self.internal_ignored[key]
        return self.internal[key]

    def __contains__(self, key):
        if hasattr(key, 'name'):
            key = key.name.lower()  # for enum elements
        return key in self.internal_ignored or key in self.internal

    def enkey_tail(self):
        """Must be the last element of the bundle for unbundling to work"""
        keys = sorted(list(self.internal.keys()))
        head = gr.HTML(elem_id="zzz_depthmap_enkey", value="\u222F" + "\u222F".join(keys), visible=False)
        return head

    def enkey_body(self):
        """This is what should be passed into the function that is called by gradio"""
        return [self.internal[x] for x in sorted(list(self.internal.keys()))]

    def add_rule(self, first, rule, second):
        first = self[first] if first in self else first
        second = self[second] if second in self else second
        if rule == 'visible-if-not':
            second.change(fn=lambda v: first.update(visible=not v), inputs=[second], outputs=[first])
        elif rule == 'visible-if':
            second.change(fn=lambda v: first.update(visible=v), inputs=[second], outputs=[first])
        else:
            raise Exception(f'Unknown rule type {rule}')

    @staticmethod
    def enkey_to_dict(inp):
        """Unbundle: get a dictionary with stuff after it is sent bby the gradio to the function.
        Enkey format: bunch of Gradio components,
        then a Gradio component, which value is concatination of names of the previous Gradio objects"""
        assert inp[-1].startswith("\u222F")
        ret = {}
        names = inp[-1].split("\u222F")[1:]
        assert len(names) == len(inp) - 1
        for i, name in enumerate(names):
            ret[name] = inp[i]
        return ret
