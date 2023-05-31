import json


def get_attribute_name(id):
    with open("fashionpedia/selected_attributes.json", "r") as file:
        atts = json.load(file)

        return atts[str(id)]
