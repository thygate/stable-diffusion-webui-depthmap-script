import gradio as gr
import enum

class GradioComponentBundle:
    """Allows easier transportation of massive ammount of named gradio inputs"""
    def __init__(self):
        self.internal = {}

    def append(self, thing):  # thing: (str | enum.Enum, gr.components.Component)
        if isinstance(thing, GradioComponentBundle):
            keys = list(thing.internal.keys())
            for key in keys:
                assert key not in self.internal, f"Already bundled component with name {key}."
                self.internal[key] = thing[key]
        elif isinstance(thing, tuple) and len(thing) == 2 and isinstance(thing[1], gr.components.Component):
            name = thing[0] if isinstance(thing[0], str) else thing[0].name.lower()  # .name is for Enums
            if hasattr(thing[0], 'df') and thing[0].df is not None:
                thing[1].value = thing[0].df
            assert name not in self.internal, f"Already bundled component with name {thing[0]}."
            self.internal[name] = thing[1]
        elif isinstance(thing, gr.components.Component) and thing.elem_id is not None:
            assert thing.elem_id not in self.internal, f"Already bundled component with name {thing.elem_id}."
            self.internal[thing.elem_id] = thing
        else:
            assert False, f"This object can not be bundled, {str(thing)}"

    def __iadd__(self, els):
        self.append(els)
        return self

    def __getitem__(self, key):
        """Return the gradio component elem_id"""
        if hasattr(key, 'name'):
            key = key.name.lower()  # for enum elements
        return self.internal[key]

    # def send_format(self):
    #     return set(self.internal.values())

    def enkey_tail(self):
        keys = sorted(list(self.internal.keys()))
        head = gr.HTML(elem_id="zzz_depthmap_enkey", value="\u222F" + "\u222F".join(keys), visible=False)
        return head

    def enkey_body(self):
        return [self.internal[x] for x in sorted(list(self.internal.keys()))]

    @staticmethod
    def enkey_to_dict(inp):
        """Enkey format: bunch of Gradio components,
        then a Gradio component, which value is concatination of names of the previous Gradio objects"""
        assert inp[-1].startswith("\u222F")
        ret = {}
        names = inp[-1].split("\u222F")[1:]
        assert len(names) == len(inp) - 1
        for i, name in enumerate(names):
            ret[name] = inp[i]
        return ret
