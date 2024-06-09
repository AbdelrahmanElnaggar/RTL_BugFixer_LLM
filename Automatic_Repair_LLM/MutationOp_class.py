import sys, inspect, subprocess
import os
from optparse import OptionParser
import copy
import random
import time 
from datetime import datetime
import math

# genprog: Class Rep: you need to write your own class 
# the next line can be removed after installation
#Adding this to import libraries from another folder
sys.path.insert(1,'/cirfix_benchmarks_code/prototype')
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyverilog.vparser.parser import parse, NodeNumbering
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.plyparser import ParseError
import pyverilog.vparser.ast as vast

import fitness

AST_CLASSES = []

for name, obj in inspect.getmembers(vast):
    if inspect.isclass(obj):
        AST_CLASSES.append(obj)

# print(AST_CLASSES)
# f = open("ast_classes.txt", "w+")
# for item in AST_CLASSES:
#     f.write(str(item) + "\n")
# f.close()

REPLACE_TARGETS = {} # dict from class to list of classes that are okay to substituite for the original class
for i in range(len(AST_CLASSES)):
    REPLACE_TARGETS[AST_CLASSES[i]] = []
    REPLACE_TARGETS[AST_CLASSES[i]].append(AST_CLASSES[i]) # can always replace with a node of the same type
    for j in range(len(AST_CLASSES)):
        # get the immediate parent classes of both classes, and if the parent if not Node, the two classes can be swapped
        if i != j and inspect.getmro(AST_CLASSES[i])[1] == inspect.getmro(AST_CLASSES[j])[1] and inspect.getmro(AST_CLASSES[j])[1] != vast.Node:
            REPLACE_TARGETS[AST_CLASSES[i]].append(AST_CLASSES[j])
       
# for key in REPLACE_TARGETS:
#     tmp = map(lambda x: x.__name__, REPLACE_TARGETS[key]) 
#     print("Class %s can be replaced by the following: %s" % (key.__name__, list(tmp)))
#     print()

"""
Valid targets for the delete and insert operators.
"""
DELETE_TARGETS = ["IfStatement", "NonblockingSubstitution", "BlockingSubstitution", "ForStatement", "Always", "Case", "CaseStatement", "DelayStatement", "Localparam", "Assign", "Block"]
INSERT_TARGETS = ["IfStatement", "NonblockingSubstitution", "BlockingSubstitution", "ForStatement", "Always", "Case", "CaseStatement", "DelayStatement", "Localparam", "Assign"]

TEMPLATE_MUTATIONS = { "increment_by_one": ("Identifier", "Plus"), "decrement_by_one": ("Identifier", "Minus"), 
                        "negate_equality": ("Eq", "NotEq"), "negate_inequality": ("NotEq", "Eq"), "negate_ulnot": ("Ulnot", "Ulnot"),
                        "sens_to_negedge": ("Sens", "Sens"), "sens_to_posedge": ("Sens", "Sens"), "sens_to_level": ("Sens", "Sens"), "sens_to_all": ("Sens", "Sens"),
                        "blocking_to_nonblocking": ("BlockingSubstitution", "NonblockingSubstitution"), "nonblocking_to_blocking": ("NonblockingSubstitution", "BlockingSubstitution")}
                        # "sll_to_sla": ("Sll", "Sla"), "sla_to_sll": ("Sla", "Sll"), 
                        # "srl_to_sra": ("Srl", "Sra"), "sra_to_srl": ("Sra", "Srl")}
                        # TODO: stmt to stmt in a block?
                        # TODO: empty if then somewhere? with like a random identifier for cond?
                        # TODO: use only registers for inc and dec by one?

WRITE_TO_FILE = True

GENOME_FITNESS_CACHE = {}

FITNESS_EVAL_TIMES = []

SEED = "None"
SRC_FILE = None
TEST_BENCH = None
PROJ_DIR = None
EVAL_SCRIPT = None
ORIG_FILE = ""
ORACLE = None
GENS = 5
POPSIZE = 200
RESTARTS = 1
FAULT_LOC = True
CONTROL_FLOW = True
LIMIT_TRANSITIVE_DEPENDENCY_SET = False
# TODO: Update defaults!
DEPENDENCY_SET_MAX = 5
REPLACEMENT_RATE = 1/3
DELETION_RATE = 1/3
INSERTION_RATE = 1/3
MUTATION_RATE = 1/2
CROSSOVER_RATE = 1/2
FITNESS_MODE = "outputwires"

TIME_NOW = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

SEED_CTR = 0
def inc_seed():
    global SEED_CTR
    SEED_CTR += 1
    return SEED + str(SEED_CTR)

class MutationOp(ASTCodeGenerator):

    def __init__(self, popsize, fault_loc, control_flow):
        self.numbering = NodeNumbering()
        self.popsize = popsize
        self.fault_loc = fault_loc
        self.control_flow = control_flow
        # temporary variables used for storing data for the mutation operators
        self.fault_loc_set = set()
        self.new_vars_in_fault_loc = dict()
        self.wires_brought_in = dict()
        self.implicated_lines = set() # contains the line number implicated by FL
        # self.stoplist = set()
        self.tmp_node = None 
        self.deletable_nodes = []
        self.insertable_nodes = []
        self.replaceable_nodes = []
        self.node_class_to_replace = None
        self.nodes_by_class = []
        self.stmt_nodes = []
        self.max_node_id = -1

    """ 
    Replaces the node corresponding to old_node_id with new_node.
    """
    def replace_with_node(self, ast, old_node_id, new_node):
        attr = vars(ast)
        for key in attr: # loop through all attributes of this AST
            if attr[key].__class__ in AST_CLASSES: # for each attribute that is also an AST
                if attr[key].node_id == old_node_id:
                    attr[key] = copy.deepcopy(new_node)
                    return
            elif attr[key].__class__ in [list, tuple]: # for attributes that are lists or tuples
                for i in range(len(attr[key])): # loop through each AST in that list or tuple
                    tmp = attr[key][i]
                    if tmp.__class__ in AST_CLASSES and tmp.node_id == old_node_id:
                        attr[key][i] = copy.deepcopy(new_node)
                        return

        for c in ast.children():
            if c: self.replace_with_node(c, old_node_id, new_node)
    
    """
    Deletes the node with the node_id provided, if such a node exists.
    """
    def delete_node(self, ast, node_id):
        attr = vars(ast)
        for key in attr: # loop through all attributes of this AST
            if attr[key].__class__ in AST_CLASSES: # for each attribute that is also an AST
                if attr[key].node_id == node_id and attr[key].__class__.__name__ in DELETE_TARGETS:
                    attr[key] = None
            elif attr[key].__class__ in [list, tuple]: # for attributes that are lists or tuples
                for i in range(len(attr[key])): # loop through each AST in that list or tuple
                    tmp = attr[key][i]
                    if tmp.__class__ in AST_CLASSES and tmp.node_id == node_id and tmp.__class__.__name__ in DELETE_TARGETS:
                        attr[key][i] = None

        for c in ast.children():
            if c: self.delete_node(c, node_id)
    
    """
    Inserts node with node_id after node with after_id.
    """
    def insert_stmt_node(self, ast, node, after_id): 
        if ast.__class__.__name__ == "Block":
            if after_id == ast.node_id:
                # node.show()
                # input("...")
                ast.statements.insert(0, copy.deepcopy(node))
                return
            else:
                insert_point = -1
                for i in range(len(ast.statements)):
                    stmt = ast.statements[i]
                    if stmt and stmt.node_id == after_id:
                        insert_point = i + 1
                        break
                if insert_point != -1:
                    # print(ast.statements)
                    ast.statements.insert(insert_point, copy.deepcopy(node))
                    # print(ast.statements)
                    return

        for c in ast.children():
            if c: self.insert_stmt_node(c, node, after_id)
    
    """
    Gets the node matching the node_id provided, if one exists, by storing it in the temporary node variable.
    Used by the insert and replace operators.
    """    
    def get_node_from_ast(self, ast, node_id):
        if ast.node_id == node_id:
            self.tmp_node = ast
        
        for c in ast.children():
            if c: self.get_node_from_ast(c, node_id)

    """ 
    Gets all the line numbers for the code implicated by the FL.
    """    
    def collect_lines_for_fl(self, ast):
        if ast.node_id in self.fault_loc_set:
            self.implicated_lines.add(ast.lineno)
        
        for c in ast.children():
            if c: self.collect_lines_for_fl(c)

    """
    Gets a list of all nodes that can be deleted.
    """
    def get_deletable_nodes(self, ast):
        # with fault localization, make sure that any node being deleted is also in DELETE_TARGETS 
        if self.fault_loc and len(self.fault_loc_set) > 0:
            if ast.node_id in self.fault_loc_set and ast.__class__.__name__ in DELETE_TARGETS:
                self.deletable_nodes.append(ast.node_id)
        else:
            if ast.__class__.__name__ in DELETE_TARGETS:
                self.deletable_nodes.append(ast.node_id)

        for c in ast.children():
            if c: self.get_deletable_nodes(c) 

    """
    Gets a list of all nodes that can be inserted into to a begin ... end block.
    """
    def get_insertable_nodes(self, ast):
        # with fault localization, make sure that any node being used is also in INSERT_TARGETS (to avoid inserting, e.g., overflow+1 into a block statement)
        if self.fault_loc and len(self.fault_loc_set) > 0: 
            if ast.node_id in self.fault_loc_set and ast.__class__.__name__ in INSERT_TARGETS:
                self.insertable_nodes.append(ast.node_id)
        else:        
            if ast.__class__.__name__ in INSERT_TARGETS:
                self.insertable_nodes.append(ast.node_id)

        for c in ast.children():
            if c: self.get_insertable_nodes(c) 
    
    """
    Gets the class of the node being replaced in a replace operation. 
    This class is used to find potential sources for the replacement.
    """
    def get_node_to_replace_class(self, ast, node_id):
        if ast.node_id == node_id:
            self.node_class_to_replace = ast.__class__

        for c in ast.children():
            if c: self.get_node_to_replace_class(c, node_id)
    
    """
    Gets all nodes that compatible to be replaced with a node of the given class type. 
    These nodes are potential sources for replace operations.
    """
    def get_replaceable_nodes_by_class(self, ast, node_type):
        if ast.__class__ in REPLACE_TARGETS[node_type]:
            self.replaceable_nodes.append(ast.node_id)

        for c in ast.children():
            if c: self.get_replaceable_nodes_by_class(c, node_type)
    
    """
    Gets all nodes that are of the given class type. 
    These nodes are used for applying mutation templates.
    """
    # TODO: do this only for fault loc set?
    def get_nodes_by_class(self, ast, node_type):
        if ast.__class__.__name__ == node_type:
            self.nodes_by_class.append(ast.node_id)

        for c in ast.children():
            if c: self.get_nodes_by_class(c, node_type)

    """
    Gets all nodes that are found within a begin ... end block. 
    These nodes are potential destinations for insert operations.
    """
    def get_nodes_in_block_stmt(self, ast):
        if ast.__class__.__name__ == "Block":
            if len(ast.statements) == 0: # if empty block, return the node id for the block (so that a node can be inserted into the empty block)
                self.stmt_nodes.append(ast.node_id)
            else:
                for c in ast.statements:
                    if c: self.stmt_nodes.append(c.node_id)
        
        for c in ast.children():
            if c: self.get_nodes_in_block_stmt(c)

    """
    Control dependency analysis of the given program branch.
    """
    def analyze_program_branch(self, ast, cond_list, mismatch_set, uniq_headers):
        if ast:
            if ast.__class__.__name__ == "Identifier" and (ast.name in mismatch_set or ast.name in tuple(self.new_vars_in_fault_loc.values())):
                for cond in cond_list:
                    if cond: self.add_node_and_children_to_fault_loc(cond, mismatch_set, uniq_headers, ast)

            for c in ast.children():
                self.analyze_program_branch(c, cond_list, mismatch_set, uniq_headers)

    """
    Add node and its immediate children to the fault loc set.    
    """
    def add_node_and_children_to_fault_loc(self, ast, mismatch_set, uniq_headers, parent=None):
        # if ast.__class__.__name__ == "Identifier" and ast.name in self.stoplist: return
        self.fault_loc_set.add(ast.node_id)
        if parent and parent.__class__.__name__ == "Identifier" and parent.name not in self.wires_brought_in: self.wires_brought_in[parent.name] = set()
        if ast.__class__.__name__ == "Identifier" and ast.name not in mismatch_set and ast.name not in uniq_headers: # and ast.name not in self.stoplist: 
            if not LIMIT_TRANSITIVE_DEPENDENCY_SET or len(self.wires_brought_in[parent.name]) < DEPENDENCY_SET_MAX:
                self.wires_brought_in[parent.name].add(ast.name)
                self.new_vars_in_fault_loc[ast.node_id] = ast.name
            # else:
            #     self.stoplist.add(ast.name)
        for c in ast.children():
            if c:
                self.fault_loc_set.add(c.node_id) 
                # add all children identifiers to depedency set
                if c.__class__.__name__ == "Identifier" and c.name not in mismatch_set and c.name not in uniq_headers: # and c.name not in self.stoplist: 
                    if not LIMIT_TRANSITIVE_DEPENDENCY_SET or len(self.wires_brought_in[parent.name]) < DEPENDENCY_SET_MAX: 
                        self.wires_brought_in[parent.name].add(c.name)
                        self.new_vars_in_fault_loc[c.node_id] = c.name
                    # else:
                    #     self.stoplist.add(c.name)

    """
    Given a set of output wires that mismatch with the oracle, get a list of node IDs that are potential fault localization targets.
    """
    # TODO: add decl to fault loc targets?
    def get_fault_loc_targets(self, ast, mismatch_set, uniq_headers, parent=None, include_all_subnodes=False):
        # data dependency analysis
        # if ast.__class__.__name__ == "Identifier" and ast.name in self.stoplist: return
        if ast.__class__.__name__ in ["BlockingSubstitution", "NonblockingSubstitution", "Assign"]: # for assignment statements =, <=
            if ast.left and ast.left.__class__.__name__ == "Lvalue" and ast.left.var:
                if ast.left.var.__class__.__name__ == "Identifier" and ast.left.var.name in mismatch_set: # single assignment
                    include_all_subnodes = True
                    parent = ast.left.var
                    if parent and not parent.name in self.wires_brought_in: self.wires_brought_in[parent.name] = set()
                    self.add_node_and_children_to_fault_loc(ast, mismatch_set, uniq_headers, parent)
                elif ast.left.var.__class__.__name__ == "LConcat": # l-concat / multiple assignments
                    for v in ast.left.var.list: 
                        if v.__class__.__name__ == "Identifier" and v.name in mismatch_set:
                            if not v.name in self.wires_brought_in: self.wires_brought_in[v.name] = set()
                            include_all_subnodes = True
                            parent = v
                            self.add_node_and_children_to_fault_loc(ast, mismatch_set, uniq_headers, parent)
        
        # control dependency analysis        
        elif self.control_flow and ast.__class__.__name__ == "IfStatement":
            self.analyze_program_branch(ast.true_statement, [ast.cond], mismatch_set, uniq_headers)
            self.analyze_program_branch(ast.false_statement, [ast.cond], mismatch_set, uniq_headers)
        elif self.control_flow and ast.__class__.__name__ == "CaseStatement":
            for c in ast.caselist: 
                if c: 
                    cond_list = [ast.comp]
                    if c.cond: 
                        for tmp_var in c.cond: cond_list.append(tmp_var)
                    self.analyze_program_branch(c.statement, cond_list, mismatch_set, uniq_headers)
        elif self.control_flow and ast.__class__.__name__ == "ForStatement":
            cond_list = []
            if ast.pre: cond_list.append(ast.pre)
            if ast.cond: cond_list.append(ast.cond)
            if ast.post: cond_list.append(ast.post)
            self.analyze_program_branch(ast.statement, cond_list, mismatch_set, uniq_headers)


        if include_all_subnodes: # recurisvely ensure all children of a fault loc target are also included in the fault loc set
            self.fault_loc_set.add(ast.node_id)
            if ast.__class__.__name__ == "Identifier" and ast.name not in mismatch_set and ast.name not in uniq_headers: # and ast.name not in self.stoplist:
                if parent and parent.__class__.__name__ == "Identifier":
                    if not LIMIT_TRANSITIVE_DEPENDENCY_SET or len(self.wires_brought_in[parent.name]) < DEPENDENCY_SET_MAX: 
                        self.wires_brought_in[parent.name].add(ast.name)
                        self.new_vars_in_fault_loc[ast.node_id] = ast.name
                    # else:
                    #     self.stoplist.add(ast.name)

        for c in ast.children():
            if c: self.get_fault_loc_targets(c, mismatch_set, uniq_headers, parent, include_all_subnodes)

        # TODO: for sdram_controller, control_flow + limit gives smaller fl set than no control_flow + limit. why? is this a bug?
    
    """
    The delete, insert, and replace operators to be called from outside the class.
    Note: node_id, with_id, and after_id would not be none if we are trying to regenerate AST from patch list, and would be none for a random mutation.
    """
    def delete(self, ast, patch_list, node_id=None):
        self.deletable_nodes = [] # reset deletable nodes for the next delete operation, in case previous delete returned early

        if node_id == None:
            self.get_deletable_nodes(ast) # get all nodes that can be deleted without breaking the AST / syntax
            if len(self.deletable_nodes) == 0: # if no nodes can be deleted, return without attepmting delete
                print("Delete operation not possible. Returning with no-op.")
                return patch_list, ast
            
            random.seed(inc_seed())
            node_id = random.choice(self.deletable_nodes) # choose a random node_id to delete
            print("Deleting node with id %s\n" % node_id)

        self.delete_node(ast, node_id) # delete the node corresponding to node_id
        self.numbering.renumber(ast) # renumber nodes
        self.max_node_id = self.numbering.c # reset max_node_id
        self.numbering.c = -1
        self.deletable_nodes = [] # reset deletable nodes for the next delete operation

        child_patchlist = copy.deepcopy(patch_list)
        child_patchlist.append("delete(%s)" % node_id) # update patch list

        return child_patchlist, ast
    
    def insert(self, ast, patch_list, node_id=None, after_id=None):
        self.insertable_nodes = [] # reset the temporary variables, in case previous insert returned early
        self.tmp_node = None

        if node_id == None and after_id == None:
            self.get_insertable_nodes(ast) # get all nodes with a type that is suited to insertion in block statements -> src
            self.get_nodes_in_block_stmt(ast) # get all nodes within a block statement -> dest
            if len(self.insertable_nodes) == 0 or len(self.stmt_nodes) == 0: # if no insertable nodes exist, exit gracefully
                print("Insert operation not possible. Returning with no-op.")
                return patch_list, ast
            random.seed(inc_seed())
            after_id = random.choice(self.stmt_nodes) # choose a random src and dest
            random.seed(inc_seed())
            node_id = random.choice(self.insertable_nodes)
            print("Inserting node with id %s after node with id %s\n" % (node_id, after_id))
        self.get_node_from_ast(ast, node_id) # get the node associated with the src node id
        self.insert_stmt_node(ast, self.tmp_node, after_id) # perform the insertion
        self.numbering.renumber(ast) # renumber nodes
        self.max_node_id = self.numbering.c # reset max_node_id
        self.numbering.c = -1
        
        child_patchlist = copy.deepcopy(patch_list)
        child_patchlist.append("insert(%s,%s)" % (node_id, after_id)) # update patch list

        return child_patchlist, ast
    
    def replace(self, ast, patch_list, node_id=None, with_id=None):
        self.tmp_node = None # reset the temporary variables (in case previous replace returned sooner)
        self.replaceable_nodes = []
        self.node_class_to_replace = None

        if node_id == None:
            if self.max_node_id == -1: # if max_id is not know yet, traverse the AST to find the number of nodes -- needed to pick a random id to replace
                self.numbering.renumber(ast)
                self.max_node_id = self.numbering.c
                self.numbering.c = -1 # reset the counter for numbering
            if self.fault_loc and len(self.fault_loc_set) > 0:
                random.seed(inc_seed())
                node_id = random.choice(tuple(self.fault_loc_set)) # get a fault loc target if fault localization is being used
            else:      
                random.seed(inc_seed())      
                node_id = random.randint(0,self.max_node_id) # get random node id to replace
            print("Node to replace id: %s" % node_id)

        self.get_node_to_replace_class(ast, node_id) # get the class of the node associated with the random node id
        print("Node to replace class: %s" % self.node_class_to_replace)
        if self.node_class_to_replace == None: # if the node does not exist, return with no-op
            return patch_list, ast
        
        if with_id == None:       
            self.get_replaceable_nodes_by_class(ast, self.node_class_to_replace) # get all valid nodes that have a class that could be substituted for the original node's class
            if len(self.replaceable_nodes) == 0: # if no replaceable nodes exist, exit gracefully
                print("Replace operation not possible. Returning with no-op.")
                return patch_list, ast
            print("Replaceable nodes: %s" % str(self.replaceable_nodes))
            random.seed(inc_seed())
            with_id = random.choice(self.replaceable_nodes) # get a random node id from the replaceable nodes
            print("Replacing node id %s with node id %s" % (node_id,with_id))  
        
        self.get_node_from_ast(ast, with_id) # get the node associated with with_id

        # safety guard: this could happen if crossover makes the GA think a node is actually suitable for replacement when in reality it is not....    
        if self.tmp_node.__class__ not in REPLACE_TARGETS[self.node_class_to_replace]: 
            print(self.tmp_node.__class__)
            print(REPLACE_TARGETS[self.node_class_to_replace])
            return patch_list, ast  

        self.replace_with_node(ast, node_id, self.tmp_node) # perform the replacement
        self.tmp_node = None # reset the temporary variables
        self.replaceable_nodes = []
        self.node_class_to_replace = None
        self.numbering.renumber(ast) # renumber nodes
        self.max_node_id = self.numbering.c # update max_node_id
        self.numbering.c = -1
        
        child_patchlist = copy.deepcopy(patch_list)
        child_patchlist.append("replace(%s,%s)" % (node_id, with_id)) # update patch list

        return child_patchlist, ast
    
    def weighted_template_choice(self, templates):
        random.seed(inc_seed())
        p = random.random()
        if p <= 0.3:
            random.seed(inc_seed())
            return random.choice(["increment_by_one", "decrement_by_one"])
        elif p <= 0.6:
            random.seed(inc_seed())
            return random.choice(["negate_equality", "negate_inequality", "negate_ulnot"])
        elif p <= 0.8:
            random.seed(inc_seed())
            return random.choice(["nonblocking_to_blocking", "blocking_to_nonblocking"])
        else:
            random.seed(inc_seed())
            return random.choice(["sens_to_negedge", "sens_to_posedge", "sens_to_level", "sens_to_all"])

    # TODO: make sure ast is a deepcopy
    def apply_template(self, ast, patch_list, template=None, node_id=None):
        self.tmp_node = None # reset the temporary variables, in case the previous template operator returned early
        self.nodes_by_class = []

        if template == None:
            template = self.weighted_template_choice(list(TEMPLATE_MUTATIONS.keys()))
            node_type = TEMPLATE_MUTATIONS[template][0]
            # print(template)
            # print(node_type)
            self.get_nodes_by_class(ast, node_type)
            # print(self.nodes_by_class)
            if len(self.nodes_by_class) == 0: 
                print("\nTemplate %s cannot be applied to AST. Returning with no-op." % template)
                return patch_list, ast # no-op
            random.seed(inc_seed())
            node_id = random.choice(self.nodes_by_class)
            # print(node_id)

        self.get_node_from_ast(ast, node_id)

        # safety guards: the following can be caused by crossover operations splitting a patchlist
        if self.tmp_node == None:
            print("Node with id %d does not exist. Returning with no-op." % node_id)
            return patch_list, ast # no-op
        elif not (self.tmp_node.__class__.__name__ == TEMPLATE_MUTATIONS[template][0]):
            print("Node classes do not match for template. This could have been caused by a crossover operation. Returning with no-op.")
            print("Node class was %s whereas expected class was %s..." % (self.tmp_node.__class__.__name__, TEMPLATE_MUTATIONS[template][0]))
            return patch_list, ast # no-op

        print("\nApplying template %s to node %d\nOld:" % (template, node_id))
        self.tmp_node.show()

        child_patchlist = copy.deepcopy(patch_list)
        
        if template == "increment_by_one":
            new_node = vast.Plus(copy.deepcopy(self.tmp_node), vast.IntConst(1, copy.deepcopy(self.tmp_node.lineno)), copy.deepcopy(self.tmp_node.lineno))
            new_node.node_id = node_id
        elif template == "decrement_by_one":
            new_node = vast.Minus(copy.deepcopy(self.tmp_node), vast.IntConst(1, copy.deepcopy(self.tmp_node.lineno)), copy.deepcopy(self.tmp_node.lineno))
        elif template == "negate_equality":
            new_node = vast.NotEq(copy.deepcopy(self.tmp_node.left), copy.deepcopy(self.tmp_node.right), copy.deepcopy(self.tmp_node.lineno))
        elif template == "negate_inequality":
            new_node = vast.Eq(copy.deepcopy(self.tmp_node.left), copy.deepcopy(self.tmp_node.right), copy.deepcopy(self.tmp_node.lineno))
        elif template == "negate_ulnot":
            new_node = vast.Ulnot(copy.deepcopy(self.tmp_node.right), copy.deepcopy(self.tmp_node.lineno))
        elif template == "sens_to_negedge":
            new_node = copy.deepcopy(self.tmp_node)
            new_node.type = "negedge"
        elif template == "sens_to_posedge":
            new_node = copy.deepcopy(self.tmp_node)
            new_node.type = "posedge"
        elif template == "sens_to_level":
            new_node = copy.deepcopy(self.tmp_node)
            new_node.type = "level"
        elif template == "sens_to_all":
            new_node = copy.deepcopy(self.tmp_node)
            new_node.type = "all"
        elif template == "nonblocking_to_blocking":
            new_node = vast.BlockingSubstitution(copy.deepcopy(self.tmp_node.left), copy.deepcopy(self.tmp_node.right), copy.deepcopy(self.tmp_node.ldelay), copy.deepcopy(self.tmp_node.rdelay), copy.deepcopy(self.tmp_node.lineno))
        elif template == "blocking_to_nonblocking":
            new_node = vast.NonblockingSubstitution(copy.deepcopy(self.tmp_node.left), copy.deepcopy(self.tmp_node.right), copy.deepcopy(self.tmp_node.ldelay), copy.deepcopy(self.tmp_node.rdelay), copy.deepcopy(self.tmp_node.lineno))
            
        new_node.node_id = node_id
        print("New:")
        new_node.show()
        self.replace_with_node(ast, node_id, new_node) # replace with new template node
        child_patchlist.append("template(%s,%s)" % (template, node_id))
        self.numbering.renumber(ast) # renumber nodes
        self.max_node_id = self.numbering.c # update max_node_id
        self.numbering.c = -1

        ast.show()

        self.tmp_node = None # reset the temporary variables
        self.nodes_by_class = []

        return child_patchlist, ast


    
    def get_crossover_children(self, parent_1, parent_2):
        if len(parent_1) < 1 or len(parent_2) < 1:
            return parent_1, parent_2

        random.seed(inc_seed())
        sp_1 = random.randint(0, len(parent_1))
        random.seed(inc_seed())
        sp_2 = random.randint(0, len(parent_2))

        parent_1_half_1 = copy.deepcopy(parent_1)[:sp_1]
        parent_1_half_2 = copy.deepcopy(parent_1)[sp_1:]
        parent_2_half_1 = copy.deepcopy(parent_2)[:sp_2]
        parent_2_half_2 = copy.deepcopy(parent_2)[sp_2:]

        print(parent_1, parent_2)
        print(sp_1, sp_2)
        print(parent_1_half_1, parent_1_half_2)
        print(parent_2_half_1, parent_2_half_2)

        parent_1_half_1.extend(parent_2_half_2)
        parent_2_half_1.extend(parent_1_half_2)

        print(parent_1_half_1, parent_2_half_1)

        return parent_1_half_1, parent_2_half_1 
    
    def crossover(self, ast, parent_1, parent_2):
        child_1, child_2 = self.get_crossover_children(parent_1, parent_2)

        child_1_ast = self.ast_from_patchlist(copy.deepcopy(ast), child_1)
        child_2_ast = self.ast_from_patchlist(copy.deepcopy(ast), child_2)

        return child_1, child_2, child_1_ast, child_2_ast
    
    def ast_from_patchlist(self, ast, patch_list):
        for m in patch_list:
            operator = m.split('(')[0]
            operands = m.split('(')[1].replace(')','').split(',')
            if operator == "replace":
                _, ast = self.replace(ast, patch_list, int(operands[0]), int(operands[1]))
            elif operator == "insert":
                _, ast = self.insert(ast, patch_list, int(operands[0]), int(operands[1]))
            elif operator == "delete":
                _, ast = self.delete(ast, patch_list, int(operands[0]))
            elif operator == "template":
                _, ast = self.apply_template(ast, patch_list, operands[0], int(operands[1]))
            else:
                print("Invalid operator in patch list: %s" % m)
        return ast
    
# def minimize_patch(mutation_op, ast, patch_list, codegen):
#     print("\n\nMinimizing patchlist...")
#     minimized = copy.deepcopy(patch_list)
#     # print(minimized)

#     for i in range(len(patch_list)-1, -1, -1): # iterate over the list in reverse order 
#         op = patch_list.pop(i)
#         # print(patch_list)
#         tmp_ast = mutation_op.ast_from_patchlist(copy.deepcopy(ast), patch_list)
#         f = open("minimized_%s.v" % TB_ID, "w+")
#         f.write(codegen.visit(tmp_ast))
#         f.close()
#         os.system("cp minimized_%s.v %s/minimized_%s.v" % (TB_ID, PROJ_DIR, TB_ID))

#         ff, _ = calc_candidate_fitness("minimized_%s.v" % TB_ID)
#         if ff == 1:
#             tmp = minimized.pop(i)
#             print("Removed operator: %s" % tmp)
#         else:
#             patch_list.insert(len(patch_list), op)
#             print("Removing operator %s causes a drop in fitness; inserting it back into the patchlist..." % op)
        
#         os.remove("minimized_%s.v" % TB_ID)
#         os.remove("%s/minimized_%s.v" % (PROJ_DIR, TB_ID))
#         # print(patch_list)

#     return minimized

def is_interesting(mutation_op, ast, codegen, patch_list):
    tmp_ast = mutation_op.ast_from_patchlist(copy.deepcopy(ast), patch_list)
    f = open("minimized_%s.v" % TB_ID, "w+")
    f.write(codegen.visit(tmp_ast))
    f.close()
    os.system("cp minimized_%s.v %s/minimized_%s.v" % (TB_ID, PROJ_DIR, TB_ID))

    ff, _ = calc_candidate_fitness("minimized_%s.v" % TB_ID)
    os.remove("minimized_%s.v" % TB_ID)
    os.remove("%s/minimized_%s.v" % (PROJ_DIR, TB_ID))
    if ff == 1:
        print("Patch %s still has a fitness of 1.0 --> interesting" % str(patch_list))
        return True
    else:
        print("Patch %s has a fitness < 1.0 --> not interesting" % str(patch_list))
        return False

"""
Delta debugging for patch minimization.
"""
def minimize_patch(mutation_op, ast, codegen, prefix, patch_list, suffix):
    mid = len(patch_list) // 2
    if mid == 0:
        return patch_list

    left = patch_list[:mid]
    if is_interesting(mutation_op, ast, codegen, prefix + left + suffix):
        return minimize_patch(mutation_op, ast, codegen, prefix, left, suffix)

    right = patch_list[mid:]
    if is_interesting(mutation_op, ast, codegen, prefix + right + suffix):
        return minimize_patch(mutation_op, ast, codegen, prefix, right, suffix)

    left = minimize_patch(mutation_op, ast, codegen, prefix, left, right + suffix)
    right = minimize_patch(mutation_op, ast, codegen, prefix + left, right, suffix)

    return left + right

def tournament_selection(mutation_op, codegen, orig_ast, popn):
    # Choose 5 random candidates for parent selection
    pool = copy.deepcopy(popn)
    while len(pool) > 5:
        random.seed(inc_seed())
        r = random.choice(pool)
        pool.remove(r)

    # generate ast from patchlist for each candidate, compute fitness for each candidate
    max_fitness = -1
    # max_fitness = math.inf
    best_parent_ast = orig_ast
    best_parent_patchlist = []

    for parent_patchlist in pool:
        parent_fitness = GENOME_FITNESS_CACHE[str(parent_patchlist)]

        if parent_fitness > max_fitness:
        # if parent_fitness < max_fitness:
            max_fitness = parent_fitness
            winner_patchlist = parent_patchlist
    
    winner_ast = copy.deepcopy(orig_ast)
    winner_ast = mutation_op.ast_from_patchlist(winner_ast, winner_patchlist)
    
    return copy.deepcopy(winner_patchlist), winner_ast

def calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path):
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)
    ORACLE=oracle_path
    print("Running VCS simulation")
    #os.system("cat %s" % fileName)

    t_start = time.time()

    #if "/" in fileName: fileName = fileName.split("/")[-1] # get the filename only if full path specified

    # TODO: The test bench is currently hard coded in eval_script. Do we want to change that?
    os.system("bash %s %s %s %s" % (EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR))
     # Run the bash script using the copied file in the new path
    #cmd = ["bash", EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR]
    #process = subprocess.Popen(cmd)
    #process.wait()
    
    if not os.path.exists("output_%s.txt" % TB_ID): 
        t_finish = time.time()
        print("errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        return 0, t_finish - t_start # if the code does not compile, return 0
        # return math.inf
    print("Output correctttttttttttttttttttttttttttttttttttttttttttttttttt")
    f = open(ORACLE, "r")
    oracle_lines = f.readlines()
    f.close()

    f = open("output_%s.txt" % TB_ID, "r")
    sim_lines = f.readlines()
    f.close()

    # weighting = "static"
    # f = open("weights.txt", "r")
    # weights = f.readlines()
    # f.close()

    # ff, total_possible = fitness.calculate_fitness(oracle_lines, sim_lines, weights, weighting)
    if FITNESS_MODE == "outputwires":
        ff, total_possible = fitness.calculate_fitness(oracle_lines, sim_lines, None, "")
        
        normalized_ff = ff/total_possible
        if normalized_ff < 0: normalized_ff = 0
        print("FITNESS = %f" % normalized_ff)

        # if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID) # Do we need to do this here? Does it make a difference?
        t_finish = time.time()

        return normalized_ff, t_finish - t_start
        # return fitness_v2.calculate_badness(oracle_lines, sim_lines, weights, weighting)
    elif FITNESS_MODE == "testcases": # experimental
        total_possible = len(sim_lines)
        count = 0
        for l in sim_lines:
            if "pass" in l.lower(): count += 1
        print("%d out of %d testcases pass" % (count, total_possible))

        t_finish = time.time()
        return count/total_possible, t_finish - t_start

def get_elite_parents(popn, pop_size):
    elite_size = int(5/100 * pop_size)
    elite = []
    for parent in popn:
        elite.append((parent, GENOME_FITNESS_CACHE[str(parent)]))
    elite.sort(key = lambda x: x[1])
    return elite[-elite_size:]
    # return elite[:-elite_size]

def strip_bits(bits):
    for i in range(len(bits)):
        bits[i] = bits[i].strip()
    return bits

def get_output_mismatch(test_bench,oracle_path):
    TB_ID = test_bench
    ORACLE= oracle_path
    f = open(ORACLE, "r")
    oracle = f.readlines()
    f.close()

    f = open("output_%s.txt" % TB_ID, "r")
    sim = f.readlines()
    f.close()

    diff_bits = []

    headers = strip_bits(oracle[0].split(","))

    if len(oracle) != len(sim): # if the output and oracle are not the same length, all output wires are defined to be mismatched
        diff_bits = headers[1:] # don't include time...
    else:
        for i in range(1, len(oracle)):
            clk = oracle[i].split(",")[0]
            tmp_oracle = strip_bits(oracle[i].split(",")[1:])
            tmp_sim = strip_bits(sim[i].split(",")[1:])
            
            for b in range(len(tmp_oracle)):
                if tmp_oracle[b] != tmp_sim[b]:
                    diff_bits.append(headers[b+1]) # offset by 1 since clk is also a header and is not an actual output
   
    res = set()

    for i in range(len(diff_bits)):
        tmp = diff_bits[i]
        if "[" in tmp:      
            res.add(tmp.split("[")[0])
        else:
            res.add(tmp)

    uniq_headers = set()
    for i in range(len(headers)):
        tmp = headers[i]
        if "[" in tmp:      
            uniq_headers.add(tmp.split("[")[0])
        else:
            uniq_headers.add(tmp)
        
    return res, uniq_headers

def seed_popn(ast, mutation_op, codegen, log, log_file):
    seeded = []
    start_time = time.time()
    while len(seeded) < 999:
        child, new_ast = mutation_op.apply_template(copy.deepcopy(ast), [])
        code = codegen.visit(new_ast)
        print(child)
        print(code)
        if str(child) not in GENOME_FITNESS_CACHE:
            f = open("candidate_%s.v" % TB_ID, "w+")
            f.write(code)
            f.close()

            os.system("cp candidate_%s.v %s/candidate_%s.v" % (TB_ID, PROJ_DIR, TB_ID))

            child_fitness = -1
            # re-parse the written candidate to check for syntax errors -> zero fitness if the candidate does not compile
            try:
                tmp_ast, directives = parse(["candidate_%s.v" % TB_ID])
            except ParseError:
                child_fitness = 0
            # if the child fitness was not 0, i.e. the parser did not throw syntax errors
            if child_fitness == -1: 
                child_fitness, sim_time = calc_candidate_fitness("candidate_%s.v" % TB_ID)
                global FITNESS_EVAL_TIMES
                FITNESS_EVAL_TIMES.append(sim_time)
                if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

            os.remove("candidate_%s.v" % TB_ID)
            os.remove("%s/candidate_%s.v" % (PROJ_DIR, TB_ID))
            
            GENOME_FITNESS_CACHE[str(child)] = child_fitness
            print(child_fitness)
            if log and log_file:
                log_file.write("\t%s --template_seeding--> %s\t\t%s\n" % ("[]", str(child), "{:.17g}".format(child_fitness)))

            if child_fitness == 1.0:
                print("\n######## REPAIR FOUND WHILE SEEDING INITIAL POPN ########")
                print(code)
                print(child)
                total_time = time.time() - start_time
                print("TOTAL TIME TAKEN TO FIND REPAIR = %f" % total_time)
                fitness_times = sum(FITNESS_EVAL_TIMES)
                # print("TOTAL TIME SPENT ON FITNESS EVALS = %f" % fitness_times)
                if log and log_file: 
                    log_file.write("\n\n######## REPAIR FOUND ########\n\t\t%s\n" % str(child))
                    log_file.write("TOTAL TIME TAKEN TO FIND REPAIR = %f\n" % total_time)
                
                minimized = minimize_patch(mutation_op, ast, codegen, [], child, [])
                print("\n\n")
                print("Minimized patch: %s" % str(minimized))

                if log and log_file:
                    log_file.write("Minimized patch: %s\n" % str(minimized))
                    log_file.close()

                sys.exit(1)
        else: # not a unique seed, log it anyways
            if log and log_file:
                log_file.write("\t%s --template_seeding--> %s\t\t%s\n" % ("[]", str(child), "{:.17g}".format(GENOME_FITNESS_CACHE[str(child)])))

        seeded.append(child)
        # input("...")
    print(GENOME_FITNESS_CACHE)
    print(len(GENOME_FITNESS_CACHE))
    return seeded

def extended_fl_for_study(fl_lines, delta):
    extended_fl = set()
    for i in range(max(fl_lines)+delta): # e.g. 0 thru 108
        if i in fl_lines:
            extended_fl.add(i)
        else:
            for j in range(1, delta+1):
                if i+j in fl_lines or i-j in fl_lines:
                    extended_fl.add(i)
    return extended_fl

