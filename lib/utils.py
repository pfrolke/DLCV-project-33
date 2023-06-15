import json
import lib.fashionpedia_type as fpt

fashionpedia_data = None

def get_by_id(group, id) -> fpt.FashionPedia[str]:
    att = get_fashionpedia_data()

    for item in att[group]:
        if item['id'] == id:
            return item

def get_attribute_name(id):
    return get_by_id('attributes', id)['name']
    
def get_fashionpedia_data(path = 'fashionpedia/attributes.json') -> fpt.FashionPedia:
    global fashionpedia_data
    if fashionpedia_data == None:
        print('loading data')
        with open(path) as f:
            fashionpedia_data = json.load(f)
    
    return fashionpedia_data
        
