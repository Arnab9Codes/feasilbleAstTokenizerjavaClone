import tree_sitter # parsing library
from tree_sitter import Language, Parser

from datasets import load_dataset
import pathlib

# selecting language to parse
JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
parser = Parser()
parser.set_language(JAVA_LANGUAGE)

start_points=[]
end_points=[]

#function for finding the leaf nodes while parsing the AST
def w(node):
    """
    node: takes ast.root_node to start traversing
    returns-> startpoints: starting index of each token
              endpointes: ending index of each token 
    """
    start_points = []
    end_points = []
    
    stack = [node]
    
    while stack:
        curr_node = stack.pop()
        
        if len(curr_node.children) == 0:
            start_points.append(curr_node.start_point)
            end_points.append(curr_node.end_point)
        else:
            for child in curr_node.children:
                stack.append(child)
    
    return start_points, end_points

# splitting each code_string by new line to get each line
def break_into_lines(code_sample):
    """
    code_sample: code taken as a single string
    return: each line in the corresponding code
    """
    return code_sample.split('\n')

# functions for getting tokens from a single code string
def get_tokens(code_sample, start_points, end_points):
    """
    code_sample: full code in string
    startpoints: starting index of each token
    endpointes: ending index of each token 
    """
    tokens=[]

    lines_in_code=break_into_lines(code_sample)
    
    assert len(start_points)==len(end_points), 'problem in finding the start and end points in the code'
    
    for i in range(len(start_points)):
        tokens.append(lines_in_code[start_points[i][0]][start_points[i][1]:end_points[i][1]])
        
    return tokens

# function for collecting all TOKENS from a dataset
def get_tokens_from_dataset(TOKENS, djava):
    """
    TOKENS: collect all tokens of the dataset, takes an empty TOKEN list
    dpy: the whole dataset 
    return: TOKENS
    """
    for code_string_iterator in range(1, 25000,1):#len(dpy['train']['whole_func_string'][10]
        tokens_per_file=[]
        start_points_per_file=[]
        end_points_per_file=[]

        tree = parser.parse(bytes(djava['train']['whole_func_string'][code_string_iterator], "utf8"))
        #print(dpy['train']['whole_func_string'][code_string_iterator])
        
        start_points_per_file, end_points_per_file=w(tree.root_node)

        #print(start_points_per_file, end_points_per_file)
        #print(tokens_per_file)
        tokens_per_file=get_tokens(djava['train']['whole_func_string'][code_string_iterator], start_points_per_file, end_points_per_file)
        #print(tokens_per_file)
        for token in reversed(tokens_per_file):
            #if not token.startswith("//"):
            print(token)
        #print("")
        
        #TOKENS.append(tokens_per_file)
        #if(code_string_iterator%100)==0:
        #    print(code_string_iterator)  
    
    return TOKENS

# writing tokens in a single file
def write_file(TOKENS):
    """
    TOKENS: contains token_list for each code string
            a list(tokens for all dataset) of lists(tokens for each code_string)
    """
    with open('AST_TOKENS_50K_to_100K.txt', 'w') as f:
        for tokens_per_code in TOKENS:
            f.write(' '.join(str(token) for token in tokens_per_code) + '\n')


dataset_codeSearch_java=load_dataset("code_search_net","java")
djava=dataset_codeSearch_java

TOKENS=[]
#TOKENS=get_tokens_from_dataset(TOKENS, djava)#passing the whole dataset
for i in range(25000):
    print(djava['train']['whole_func_string'][i])


