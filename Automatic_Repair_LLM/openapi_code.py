import pandas as pd
import numpy as np
import sys, inspect, subprocess
import os
from optparse import OptionParser
import copy
import random
import time 
from datetime import datetime
import math
from openai import OpenAI
from secret_key import OPENAI_API_KEY
import MutationOp_class
from MutationOp_class import MutationOp
from MutationOp_class import get_output_mismatch, calc_candidate_fitness,extended_fl_for_study
import argparse
import json
import tiktoken
import shutil
from itertools import islice
import re

#Adding this to import libraries from another folder
sys.path.insert(1,'/cirfix_benchmarks_code/prototype')

from pyverilog.vparser.parser import parse, NodeNumbering
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.plyparser import ParseError
from pyverilog.vparser.ast import Node
import pyverilog.vparser.ast as vast
import fitness


#print(num_tokens_from_string("Hello world, let's test tiktoken.", "gpt-3.5-turbo"))


output_file_path = f"output_test.v"
output_directory = '/Automatic_Repair_LLM/'
output_prompt_path=output_directory+"prompt"

if not os.path.exists(output_prompt_path):
    os.makedirs(output_prompt_path)
    print(f"Directory '{output_prompt_path}' created.")
else:
    print(f"Directory '{output_prompt_path}' already exists.")

jason_file_path = 'output_data.json'
pandas_save_result_path = '/Automatic_Repair_LLM/save_result.csv'
pandas_save_feedback_result_path = '/Automatic_Repair_LLM/save_feedback_result.csv'
#baseline_save_results = '/Automatic_Repair_LLM/save_baseline_result.csv'

# used in feedback
context = [ {'role':'system', 'content':"You are a helpful assistant for fixing Verilog and system Verilog code."} ]  # accumulate messages




def directory_creation(scenario_ID,experiment_number):
    
    #directory_path=output_directory+"SC_ID_"+str(scenario_ID)+"/"+f_name
    directory_path=output_directory+"Experimental_output/"
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"exp_num_"+str(experiment_number)+"/"
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    
    directory_path=directory_path+"SC_ID_"+str(scenario_ID)+"/"
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        

    return directory_path
        

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_string_gpto(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def basic_scenario(file_name,buggy_src_file_path,directory_path):

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt_basic'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()
    #prompt= f" fix the following buggy verilog code: \n {buggy_verilog_code}\n. generate only code without any extra words or explanation"   
    
    #prompt= f" Generate a fix for the following buggy verilog code without generating any extra words or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Generate a fix for the following buggy verilog code without generating any comments or explanation : \n {buggy_verilog_code}\n"
    prompt= f" Give me the full working code only without generating any extra words or comments or explanation to your answer, Generate a fix for the following buggy verilog code : \n {buggy_verilog_code}\n"
    prompt2= f""" 

        Your Task: 

        ----------------

        Correct a buggy Verilog code provided below. You'll be given: 

        1. The buggy Verilog code. 

        Instructions: 

        ----------------

        Deliver a complete, functional code without additional words, comments, or explanations. 

        Focus on making one or more simple changes within the code, targeting the lines where the bug could originate. 

        Do not change the overall logic and structure of the code. 


        Please find the provided buggy Verilog code below: 

        ---------------- 

        {buggy_verilog_code} 

        """ 
    
    
    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt2)) 
    #return prompt
    return prompt2
    
def bug_description_scenario(file_name,buggy_src_file_path,directory_path,bug_description):

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt_basic'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()
    #prompt= f" fix the following buggy verilog code: \n {buggy_verilog_code}\n. generate only code without any extra words or explanation"   
    
    #prompt= f" Generate a fix for the following buggy verilog code without generating any extra words or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Generate a fix for the following buggy verilog code without generating any comments or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Give me the full working code only without generating any extra words or comments or explanation to your answer, Generate a fix for the following buggy verilog code : \n {buggy_verilog_code} ,given that the bug description is {bug_description} \n"
    #prompt = f"Provide the complete functioning code without adding extra words, comments, or explanations. Fix the given buggy Verilog code:\n{buggy_verilog_code}, considering the bug description: {bug_description}\n"
    prompt1 = f"Provide the complete functioning code without adding extra words, comments, or explanations to your answer. Given the bug description: {bug_description}, fix the following buggy Verilog code:\n{buggy_verilog_code}\n"
    
    prompt2=f"""Your task is to correct the specified bug in the given Verilog code while maintaining its functionality. The description of the bug is <<< {bug_description}>>>. It is important to provide the complete functioning code without adding extra words, comments, or explanations to your response. Your goal is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should not alter the overall logic of the code. The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>> Your task is to identify and implement the necessary correction(s) to resolve the bug while adhering to the specified guidelines. """
    
    prompt3=f""" Your task is to provide the complete functioning code as a hardware engineer without adding extra words, comments, or explanations to your response. Your goal is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should not alter the overall logic of the code. The description of the bug is <<< {bug_description}>>>. The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>>  """
    
    prompt4= f""" 

        Your Task: 

        ----------------

        Correct a buggy Verilog code provided below. You'll be given: 

        1. The buggy Verilog code. 

        2. A description of the bug. 

        Instructions: 

        ----------------

        Deliver a complete, functional code without additional words, comments, or explanations. 

        Focus on making one or more simple changes within the code, targeting the lines where the bug could originate. 

        Do not change the overall logic and structure of the code. 


        Bug Description: 

        ----------------


        '{bug_description}' 


        Please find the provided buggy Verilog code below: 

        ---------------- 

        {buggy_verilog_code} 

        """ 


    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt4)) 
    #return prompt
    return prompt4


def scenario_mismatch(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description):
    

    optparser = OptionParser()
    optparser.add_option("-v","--version",action="store_true",dest="showversion",
                         default=False,help="Show the version")
    optparser.add_option("-I","--include",dest="include",action="append",
                         default=[],help="Include path")
    optparser.add_option("-D",dest="define",action="append",
                         default=[],help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = [buggy_src_file_path, test_bench_path]
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")

    for f in filelist:
        if not os.path.exists(f): raise IOError("file not found: " + f)

    codegen = ASTCodeGenerator()
    # parse the files (in filelist) to ASTs (PyVerilog ast)
    ast, directives = parse([buggy_src_file_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    
    #ast, _ = parse([buggy_src_file_path])

    ast.show()
    print(ast)
    src_code = codegen.visit(ast)
    print(src_code)

    print("\n\n")

    mutation_op = MutationOp(0, True, True)
    orig_fitness, sim_time = calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path)
    print("orig_fitness = ")
    print(orig_fitness)
    
    mismatch_set, uniq_headers = get_output_mismatch(TB_ID,oracle_path)
    print(mismatch_set)
    
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

    comp_failures = 0
    
    if mutation_op.fault_loc:
        tmp_mismatch_set = copy.deepcopy(mismatch_set)
        print()
        mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers) # compute fault localization for the parent
        print("Initial Fault Localization:", str(mutation_op.fault_loc_set))
        while len(mutation_op.new_vars_in_fault_loc) > 0:
            new_mismatch_set = set(mutation_op.new_vars_in_fault_loc.values())
            print("New vars in fault loc:", new_mismatch_set)
            mutation_op.new_vars_in_fault_loc = dict()
            tmp_mismatch_set = tmp_mismatch_set.union(new_mismatch_set)
            mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers)
            print("Fault Localization:", str(mutation_op.fault_loc_set))
        print("Final mismatch set:", tmp_mismatch_set)
        print("Final Fault Localization:", str(mutation_op.fault_loc_set))
        print(len(mutation_op.fault_loc_set))
        # print(mutation_op.stoplist)
        # print(mutation_op.wires_brought_in)
        
                # exit(1)

        mutation_op.implicated_lines = set()
        mutation_op.collect_lines_for_fl(ast)
        print("Lines implicated by FL: %s" % str(mutation_op.implicated_lines))
        print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_lines))

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt_mismatch'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()
    #prompt= f" fix the following buggy verilog code: \n {buggy_verilog_code}\n. generate only code without any extra words or explanation"   
    
    #prompt= f" Generate a fix for the following buggy verilog code without generating any extra words or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Generate a fix for the following buggy verilog code without generating any comments or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Give me the full working code only without generating any extra words or comments or explanation to your answer, Generate a fix for the following buggy verilog code : \n {buggy_verilog_code}\n"
    prompt1 = f"Provide the complete functioning code without adding extra words, comments, or explanations to your answer. Given the bug description: {bug_description}, the bug might be originating from one or more of the elements listed in this mismatch list:\n{tmp_mismatch_set}\n, fix the following buggy Verilog code:\n{buggy_verilog_code}\n"
    prompt2=f"""
        Your task is to correct the specified bug in the given Verilog code while maintaining its functionality. The description of the bug is <<< {bug_description}>>>. After comparing the output of the correct and buggy code, a list of signals or variables or registers causing the bug is obtained, which is <<<{tmp_mismatch_set}>>>. This list signifies the mismatch between the correct and buggy code, suggesting that the bug may be due to one or more elements in this list. 
        It is important to provide the complete functioning code without adding extra words, comments, or explanations to your response. Your goal is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should not alter the overall logic of the code. The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>> Your task is to identify and implement the necessary correction(s) to resolve the bug while adhering to the specified guidelines. 
            """
    prompt3=f""" Your task is to provide the complete functioning code as a hardware engineer without adding extra words, comments, or explanations to your response. Your goal is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should not alter the overall logic of the code. The description of the bug is <<< {bug_description}>>>. After comparing the output of the correct and buggy code, a list of signals or variables or registers causing the bug is obtained in the following list <<<{tmp_mismatch_set}>>>. This list signifies the mismatch between the correct and buggy code, suggesting that the bug may be due to one or more elements in this list.  
        The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>>  """
    
    prompt4= f""" 

        Your Task: 
        ----------------
        Correct a buggy Verilog code provided below. You'll be given: 

        1. The buggy Verilog code. 
        2. A description of the bug. 
        3. A list of potential elements where the bug might originate. 

        Instructions: 
        ----------------

        Deliver a complete, functional code without additional words, comments, or explanations. 
        Focus on making one or more simple changes within the code, targeting the lines where the bug could originate. 
        Do not change the overall logic and structure of the code. 


        Bug Description: 
        ----------------

        '{bug_description}' 

        Identified Differences: 
        ---------------- 

        After comparing the output of the correct and buggy code, we have identified elements that may be contributing to the bug.
        These elements, including Input Ports, Output Ports, Registers, Wires, or Data Types, will be listed below: 

        {tmp_mismatch_set} 
        

        Please find the provided buggy Verilog code below: 
        ---------------- 

        {buggy_verilog_code} 

        """ 
    prompt6= f""" 

        Your Task: 
        ----------------
        Correct a buggy Verilog code provided below. You'll be given: 

        1. The buggy Verilog code. 
        2. A description of the bug. 
        3. A list of potential elements where the bug might originate. 

        Instructions: 
        ----------------
        
        Start by listing the lines of code from the provided mismatch list that might be causing errors. Then, analyze which of these lines need to be corrected.
        Focus on making one or more simple changes within the code, targeting the lines where the bug could originate. Ensure not to change the overall logic and structure of the code.
        Deliver a complete, functional code without additional words, comments, or explanations. Return the complete, corrected code between the following pattern: start_code and end_code.

    

        Bug Description: 
        ----------------

        '{bug_description}' 

        Identified Differences: 
        ---------------- 

        After comparing the output of the correct and buggy code, we have identified elements that may be contributing to the bug. These elements, including Input Ports, Output Ports, Registers, Wires, or Data Types, will be listed below: 

        {tmp_mismatch_set} 
        

        Please find the provided buggy Verilog code below: 
        ---------------- 

        {buggy_verilog_code} 

        """ 
    

    print("prompt =")
    print(prompt4)
    
    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt4)) 
    #return prompt
    return prompt4


def extract_fixed_code_from_json(json_text):
    try:
        data = json.loads(json_text)
        if 'fixed_code' in data:
            return data['fixed_code']
        else:
            print("Error: 'fixed_code' key not found in the JSON data.")
            return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
        return None

def scenario_tech1(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description):
    

    optparser = OptionParser()
    optparser.add_option("-v","--version",action="store_true",dest="showversion",
                         default=False,help="Show the version")
    optparser.add_option("-I","--include",dest="include",action="append",
                         default=[],help="Include path")
    optparser.add_option("-D",dest="define",action="append",
                         default=[],help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = [buggy_src_file_path, test_bench_path]
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")

    for f in filelist:
        if not os.path.exists(f): raise IOError("file not found: " + f)

    codegen = ASTCodeGenerator()
    # parse the files (in filelist) to ASTs (PyVerilog ast)
    ast, directives = parse([buggy_src_file_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    
    #ast, _ = parse([buggy_src_file_path])

    ast.show()
    print(ast)
    src_code = codegen.visit(ast)
    print(src_code)

    print("\n\n")

    mutation_op = MutationOp(0, True, True)
    orig_fitness, sim_time = calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path)
    print("orig_fitness = ")
    print(orig_fitness)
    
    mismatch_set, uniq_headers = get_output_mismatch(TB_ID,oracle_path)
    print(mismatch_set)
    
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

    comp_failures = 0
    
    if mutation_op.fault_loc:
        tmp_mismatch_set = copy.deepcopy(mismatch_set)
        print()
        mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers) # compute fault localization for the parent
        print("Initial Fault Localization:", str(mutation_op.fault_loc_set))
        while len(mutation_op.new_vars_in_fault_loc) > 0:
            new_mismatch_set = set(mutation_op.new_vars_in_fault_loc.values())
            print("New vars in fault loc:", new_mismatch_set)
            mutation_op.new_vars_in_fault_loc = dict()
            tmp_mismatch_set = tmp_mismatch_set.union(new_mismatch_set)
            mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers)
            print("Fault Localization:", str(mutation_op.fault_loc_set))
        print("Final mismatch set:", tmp_mismatch_set)
        print("Final Fault Localization:", str(mutation_op.fault_loc_set))
        print(len(mutation_op.fault_loc_set))
        # print(mutation_op.stoplist)
        # print(mutation_op.wires_brought_in)
        
                # exit(1)

        mutation_op.implicated_lines = set()
        mutation_op.collect_lines_for_fl(ast)
        print("Lines implicated by FL: %s" % str(mutation_op.implicated_lines))
        print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_lines))

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt_tech1'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()
    #prompt= f" fix the following buggy verilog code: \n {buggy_verilog_code}\n. generate only code without any extra words or explanation"   
    
    #prompt= f" Generate a fix for the following buggy verilog code without generating any extra words or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Generate a fix for the following buggy verilog code without generating any comments or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Give me the full working code only without generating any extra words or comments or explanation to your answer, Generate a fix for the following buggy verilog code : \n {buggy_verilog_code}\n"
    prompt1 = f"Provide the complete functioning code without adding extra words, comments, or explanations to your answer. Given the bug description: {bug_description}, the bug might be originating from one or more of the elements listed in this mismatch list:\n{tmp_mismatch_set}\n, fix the following buggy Verilog code:\n{buggy_verilog_code}\n"
    prompt2=f"""
        Your task is to correct the specified bug in the given Verilog code while maintaining its functionality. The description of the bug is <<< {bug_description}>>>. After comparing the output of the correct and buggy code, a list of signals or variables or registers causing the bug is obtained, which is <<<{tmp_mismatch_set}>>>. This list signifies the mismatch between the correct and buggy code, suggesting that the bug may be due to one or more elements in this list. 
        It is important to provide the complete functioning code without adding extra words, comments, or explanations to your response. Your goal is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should not alter the overall logic of the code. The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>> Your task is to identify and implement the necessary correction(s) to resolve the bug while adhering to the specified guidelines. 
            """
    prompt3=f""" Your task is to provide the complete functioning code as a hardware engineer without adding extra words, comments, or explanations to your response. Your goal is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should not alter the overall logic of the code. The description of the bug is <<< {bug_description}>>>. After comparing the output of the correct and buggy code, a list of signals or variables or registers causing the bug is obtained in the following list <<<{tmp_mismatch_set}>>>. This list signifies the mismatch between the correct and buggy code, suggesting that the bug may be due to one or more elements in this list.  
        The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>>  """
    
    prompt4= f""" 

        Your Task: 
        ----------------
        Correct a buggy Verilog code provided below. You'll be given: 

        1. The buggy Verilog code. 
        2. A description of the bug. 
        3. A list of potential elements where the bug might originate. 

        Instructions: 
        ----------------
        Think step by step. Focus on making one or more simple changes within the code, targeting the lines where the bug could originate.
        Provide reasoning for every code statement you generate. Ensure not to change the overall logic and structure of the code
        Finally, return the complete, functional code between the following pattern: start_code and end_code, ensuring there are no additional words, comments, or explanations.

        



        Bug Description: 
        ----------------

        '{bug_description}' 

        Identified Differences: 
        ---------------- 

        After comparing the output of the correct and buggy code, we have identified elements that may be contributing to the bug. These elements, including Input Ports, Output Ports, Registers, Wires, or Data Types, will be listed below: 

        {tmp_mismatch_set} 
        

        Please find the provided buggy Verilog code below: 
        ---------------- 

        {buggy_verilog_code} 

        """ 
    
    prompt5= f""" 

        Your Task: 
        ----------------
        Correct a buggy Verilog code provided below. You'll be given: 

        1. The buggy Verilog code. 
        2. A description of the bug. 
        
        Instructions: 
        ----------------
        Think step by step. Focus on making one or more simple changes within the code, targeting the lines where the bug could originate.
        Provide reasoning for every code statement you generate. Ensure not to change the overall logic and structure of the code
        Finally, return the complete, functional code between the following pattern: start_code and end_code, ensuring there are no additional words, comments, or explanations.


        Bug Description: 
        ----------------

        '{bug_description}' 

        Please find the provided buggy Verilog code below: 
        ---------------- 

        {buggy_verilog_code} 

        """ 
    

    print("prompt =")
    print(prompt5)
    
    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt5)) 
    #return prompt
    return prompt5



def scenario_tech2(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description):
    

    optparser = OptionParser()
    optparser.add_option("-v","--version",action="store_true",dest="showversion",
                         default=False,help="Show the version")
    optparser.add_option("-I","--include",dest="include",action="append",
                         default=[],help="Include path")
    optparser.add_option("-D",dest="define",action="append",
                         default=[],help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = [buggy_src_file_path, test_bench_path]
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")

    for f in filelist:
        if not os.path.exists(f): raise IOError("file not found: " + f)

    codegen = ASTCodeGenerator()
    # parse the files (in filelist) to ASTs (PyVerilog ast)
    ast, directives = parse([buggy_src_file_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    
    #ast, _ = parse([buggy_src_file_path])

    ast.show()
    print(ast)
    src_code = codegen.visit(ast)
    print(src_code)

    print("\n\n")

    mutation_op = MutationOp(0, True, True)
    orig_fitness, sim_time = calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path)
    print("orig_fitness = ")
    print(orig_fitness)
    
    mismatch_set, uniq_headers = get_output_mismatch(TB_ID,oracle_path)
    print(mismatch_set)
    
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

    comp_failures = 0
    
    if mutation_op.fault_loc:
        tmp_mismatch_set = copy.deepcopy(mismatch_set)
        print()
        mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers) # compute fault localization for the parent
        print("Initial Fault Localization:", str(mutation_op.fault_loc_set))
        while len(mutation_op.new_vars_in_fault_loc) > 0:
            new_mismatch_set = set(mutation_op.new_vars_in_fault_loc.values())
            print("New vars in fault loc:", new_mismatch_set)
            mutation_op.new_vars_in_fault_loc = dict()
            tmp_mismatch_set = tmp_mismatch_set.union(new_mismatch_set)
            mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers)
            print("Fault Localization:", str(mutation_op.fault_loc_set))
        print("Final mismatch set:", tmp_mismatch_set)
        print("Final Fault Localization:", str(mutation_op.fault_loc_set))
        print(len(mutation_op.fault_loc_set))
        # print(mutation_op.stoplist)
        # print(mutation_op.wires_brought_in)
        
                # exit(1)

        mutation_op.implicated_lines = set()
        mutation_op.collect_lines_for_fl(ast)
        print("Lines implicated by FL: %s" % str(mutation_op.implicated_lines))
        print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_lines))

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt_tech1'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()
    #prompt= f" fix the following buggy verilog code: \n {buggy_verilog_code}\n. generate only code without any extra words or explanation"   
    
    #prompt= f" Generate a fix for the following buggy verilog code without generating any extra words or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Generate a fix for the following buggy verilog code without generating any comments or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Give me the full working code only without generating any extra words or comments or explanation to your answer, Generate a fix for the following buggy verilog code : \n {buggy_verilog_code}\n"
    prompt1 = f"Provide the complete functioning code without adding extra words, comments, or explanations to your answer. Given the bug description: {bug_description}, the bug might be originating from one or more of the elements listed in this mismatch list:\n{tmp_mismatch_set}\n, fix the following buggy Verilog code:\n{buggy_verilog_code}\n"
    prompt2=f"""
        Your task is to correct the specified bug in the given Verilog code while maintaining its functionality. The description of the bug is <<< {bug_description}>>>. After comparing the output of the correct and buggy code, a list of signals or variables or registers causing the bug is obtained, which is <<<{tmp_mismatch_set}>>>. This list signifies the mismatch between the correct and buggy code, suggesting that the bug may be due to one or more elements in this list. 
        It is important to provide the complete functioning code without adding extra words, comments, or explanations to your response. Your goal is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should not alter the overall logic of the code. The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>> Your task is to identify and implement the necessary correction(s) to resolve the bug while adhering to the specified guidelines. 
            """
    prompt3=f""" Your task is to provide the complete functioning code as a hardware engineer without adding extra words, comments, or explanations to your response. Your goal is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should not alter the overall logic of the code. The description of the bug is <<< {bug_description}>>>. After comparing the output of the correct and buggy code, a list of signals or variables or registers causing the bug is obtained in the following list <<<{tmp_mismatch_set}>>>. This list signifies the mismatch between the correct and buggy code, suggesting that the bug may be due to one or more elements in this list.  
        The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>>  """
    
    prompt4= f""" 

        Your Task: 
        ----------------
        Correct a buggy Verilog code provided below. You'll be given: 

        1. The buggy Verilog code. 
        2. A description of the bug. 
        3. A list of potential elements where the bug might originate. 

        Instructions: 
        ----------------
        
        Start by listing the lines of code from the provided mismatch list that might be causing errors. Then, analyze which of these lines need to be corrected.
        Focus on making one or more simple changes within the code, targeting the lines where the bug could originate. Ensure not to change the overall logic and structure of the code.
        Deliver a complete, functional code without additional words, comments, or explanations. Return the complete, corrected code between the following pattern: start_code and end_code.

        



        Bug Description: 
        ----------------

        '{bug_description}' 

        Identified Differences: 
        ---------------- 

        After comparing the output of the correct and buggy code, we have identified elements that may be contributing to the bug. These elements, including Input Ports, Output Ports, Registers, Wires, or Data Types, will be listed below: 

        {tmp_mismatch_set} 
        

        Please find the provided buggy Verilog code below: 
        ---------------- 

        {buggy_verilog_code} 

        """ 
    

    print("prompt =")
    print(prompt4)
    
    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt4)) 
    #return prompt
    return prompt4




def scenario_tech3(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description):
    

    optparser = OptionParser()
    optparser.add_option("-v","--version",action="store_true",dest="showversion",
                         default=False,help="Show the version")
    optparser.add_option("-I","--include",dest="include",action="append",
                         default=[],help="Include path")
    optparser.add_option("-D",dest="define",action="append",
                         default=[],help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = [buggy_src_file_path, test_bench_path]
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")

    for f in filelist:
        if not os.path.exists(f): raise IOError("file not found: " + f)

    codegen = ASTCodeGenerator()
    # parse the files (in filelist) to ASTs (PyVerilog ast)
    ast, directives = parse([buggy_src_file_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    
    #ast, _ = parse([buggy_src_file_path])

    ast.show()
    print(ast)
    src_code = codegen.visit(ast)
    print(src_code)

    print("\n\n")

    mutation_op = MutationOp(0, True, True)
    orig_fitness, sim_time = calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path)
    print("orig_fitness = ")
    print(orig_fitness)
    
    mismatch_set, uniq_headers = get_output_mismatch(TB_ID,oracle_path)
    print(mismatch_set)
    
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

    comp_failures = 0
    
    if mutation_op.fault_loc:
        tmp_mismatch_set = copy.deepcopy(mismatch_set)
        print()
        mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers) # compute fault localization for the parent
        print("Initial Fault Localization:", str(mutation_op.fault_loc_set))
        while len(mutation_op.new_vars_in_fault_loc) > 0:
            new_mismatch_set = set(mutation_op.new_vars_in_fault_loc.values())
            print("New vars in fault loc:", new_mismatch_set)
            mutation_op.new_vars_in_fault_loc = dict()
            tmp_mismatch_set = tmp_mismatch_set.union(new_mismatch_set)
            mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers)
            print("Fault Localization:", str(mutation_op.fault_loc_set))
        print("Final mismatch set:", tmp_mismatch_set)
        print("Final Fault Localization:", str(mutation_op.fault_loc_set))
        print(len(mutation_op.fault_loc_set))
        # print(mutation_op.stoplist)
        # print(mutation_op.wires_brought_in)
        
                # exit(1)

        mutation_op.implicated_lines = set()
        mutation_op.collect_lines_for_fl(ast)
        print("Lines implicated by FL: %s" % str(mutation_op.implicated_lines))
        print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_lines))

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt_tech3'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()
    #prompt= f" fix the following buggy verilog code: \n {buggy_verilog_code}\n. generate only code without any extra words or explanation"   
    
    #prompt= f" Generate a fix for the following buggy verilog code without generating any extra words or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Generate a fix for the following buggy verilog code without generating any comments or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Give me the full working code only without generating any extra words or comments or explanation to your answer, Generate a fix for the following buggy verilog code : \n {buggy_verilog_code}\n"
    prompt1_old= f"""
        Provide a complete, functional Verilog code without adding extra words, comments, or explanations. Your task is to fix the buggy Verilog code provided below based on the bug description: {bug_description}. Make one or more simple changes within the code, targeting the lines where the bug could originate. Do not change the overall logic and structure of the code. Return the corrected code between the following pattern: start_code and end_code. 
        Follow these repair guidelines based on the bug description: If there's a defect in conditional statements, consider inverting the condition of the code block. If the issue lies in the sensitivity list, trigger an always block on the following: Signal's falling edge or signal's rising edge or any change to a variable within the block or Signal's level change. 
        When dealing with assignment block defects, consider changing a blocking assignment to nonblocking or converting a non-blocking assignment to blocking. For numeric value discrepancies, adjust the value of an identifier by either incrementing or decrementing it by 1. 
               """
    

    prompt1=f""" Provide the complete functioning code without adding extra words, comments, or explanations to your answer. Given the bug description: <<<{bug_description}>>>, fix the following buggy Verilog code:\n<<<{buggy_verilog_code}>>>\n 
                Follow these repair guidelines based on the bug description: If the issue is with conditional statements, try flipping the condition. If it's about sensitivity, make sure to set triggers for when the signal goes up or down, when any variable changes, or when the signal's level changes. For assignment problems, switch between blocking and non-blocking assignments. And if there's a number problem, adjust the value by adding or subtracting 1 """
    
    prompt2=f"""
            Your task is to correct the specified bug in the given Verilog code while maintaining its functionality. The description of the bug is <<< {bug_description}>>>. To accomplish this, deliver a complete, functional code without additional words, comments, or explanations. Focus on making one or more simple changes within the code, targeting the lines where the bug might originate. However, it's crucial not to change the overall logic and structure of the code. The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>> Your goal is to identify and implement the necessary correction(s) to resolve the bug while adhering to the specified guidelines. 
            Return the complete, corrected code between the following pattern: start_code and end_code. Follow these repair guidelines based on the bug description: 
            If there's a defect in conditional statements, consider inverting the condition of the code block. 
            If the issue lies in the sensitivity list, trigger an always block on the following: Signal's falling edge, signal's rising edge, any change to a variable within the block, or signal's level change. 
            When dealing with assignment block defects, consider changing a blocking assignment to nonblocking or converting a non-blocking assignment to blocking. 
            For numeric value discrepancies, adjust the value of an identifier by either incrementing or decrementing it by 1. 

            """
    
    prompt3=f""" Your task as a hardware engineer assistant is to provide the complete functioning Verilog code, addressing a specific bug without adding extra words, comments, or explanations to your response. Your objective is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should refrain from altering the overall logic of the code. Ensure the corrected code is returned between the following pattern: start_code and end_code. 
                Follow these repair guidelines based on the bug description: 
                If there's a defect in conditional statements, consider inverting the condition of the code block. 
                If the issue lies in the sensitivity list, verify that the always block triggers appropriately. This includes triggering the always block on the following conditions: when the signal falls or rises, when there is any change to a variable within the block, or when the signal's level changes. 
                When dealing with assignment block defects, consider changing a blocking assignment to nonblocking or converting a non-blocking assignment to blocking. 
                For numeric value discrepancies, adjust the value of an identifier by either incrementing or decrementing it by 1. 
                The bug description is as follows: <<< {bug_description} >>>. Below is the provided buggy Verilog code:  <<< \n{buggy_verilog_code}\n >>> """
    
    
    prompt3_old=f""" Your task as a hardware engineer assistant is to provide the complete functioning Verilog code, addressing a specific bug without adding extra words, comments, or explanations to your response. Your objective is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should refrain from altering the overall logic of the code. The bug description is as follows: <<< {bug_description} >>>. Below is the provided buggy Verilog code:  <<< \n{buggy_verilog_code}\n >>>
                Ensure the corrected code is returned between the following pattern: start_code and end_code. Follow these repair guidelines based on the bug description: 
                If there's a defect in conditional statements, consider inverting the condition of the code block. 
                If the issue lies in the sensitivity list, verify that the always block triggers appropriately. This includes triggering the always block on the following conditions: when the signal falls or rises, when there is any change to a variable within the block, or when the signal's level changes. 
                When dealing with assignment block defects, consider changing a blocking assignment to nonblocking or converting a non-blocking assignment to blocking. 
                For numeric value discrepancies, adjust the value of an identifier by either incrementing or decrementing it by 1.  """
    
    '''
    Verilog Repair Guideline:
    1. Conditional Statements:
    If there's a defect in conditional statements, invert the condition of the code block.
    2. Sensitivity List:
    If the issue lies in the sensitivity list:
    Trigger an always block on:
    Signal's falling edge.
    Signal's rising edge.
    Any change to a variable within the block.
    Signal's level change.
    3. Assignment Block:
    When dealing with assignment block defects:
    Change a blocking assignment to nonblocking.
    Convert a non-blocking assignment to blocking.
    4. Numeric Value:
    For numeric value discrepancies:
    Increment the value of an identifier by 1.
    Decrement the value of an identifier by 1.
    5. Additional Transformations:
    Invert equality.
    Invert inequality.
    Invert ULNOT.
    Switch nonblocking assignments to blocking.
    Switch blocking assignments to nonblocking.
    Change sensitivity to negative edge.
    Change sensitivity to positive edge.
    Change sensitivity to level.
    Change sensitivity to all.
    Following these steps systematically will help in identifying and rectifying defects in Verilog code efficiently.
    '''
    '''
     Follow the following guidelines for repair: 
        if the defect is in conditional statements then negate the condition of code block 
        if the defect is in sensitivity list, consider one of the following solutions:
        Trigger an always block on a signals falling edge, Trigger an always block on a signals rising edge, Trigger an always block on any change to a variable within the block, Trigger an always block when a signal is level
        if the defect in assignment block, consider the following:
        Change a blocking assignment to nonblocking, Change a non-blocking assignment to blocking 
        if defect in numeric value, perform the following:
        Increment the value of an identifier by 1, Decrement the value of an identifier by 1
    '''


    prompt4_old= f""" 

        Your Task: 
        ----------------
        Correct a buggy Verilog code provided below. You'll be given: 

        1. The buggy Verilog code. 
        2. A description of the bug. 
        3. A list of potential elements where the bug might originate. 

        Instructions: 
        ----------------
        Deliver a complete, functional code without additional words, comments, or explanations. 
        Focus on making one or more simple changes within the code, targeting the lines where the bug could originate. 
        Do not change the overall logic and structure of the code.
        Return the complete, corrected code between the following pattern: start_code and end_code.
        Based on the bug description follow the following guidelines for repair: 
        1. Conditional Statements:
        If there's a defect in conditional statements, invert the condition of the code block.
        2. Sensitivity List:
        If the issue lies in the sensitivity list:
        Trigger an always block on:
        Signal's falling edge.
        Signal's rising edge.
        Any change to a variable within the block.
        Signal's level change.
        3. Assignment Block:
        When dealing with assignment block defects:
        Change a blocking assignment to nonblocking.
        Convert a non-blocking assignment to blocking.
        4. Numeric Value:
        For numeric value discrepancies:
        Increment the value of an identifier by 1.
        Decrement the value of an identifier by 1.



        Bug Description: 
        ----------------

        '{bug_description}' 

        Identified Differences: 
        ---------------- 

        After comparing the output of the correct and buggy code, we have identified elements that may be contributing to the bug. These elements, including Input Ports, Output Ports, Registers, Wires, or Data Types, will be listed below: 

        {tmp_mismatch_set} 
        

        Please find the provided buggy Verilog code below: 
        ---------------- 

        {buggy_verilog_code} 

        """ 
    
    prompt4= f""" 

        Your Task: 
        ----------------
        Correct a buggy Verilog code provided below. You'll be given: 

        1. The buggy Verilog code. 
        2. A description of the bug. 

        Instructions: 
        ----------------
        Deliver a complete, functional code without additional words, comments, or explanations. 
        Focus on making one or more simple changes within the code, targeting the lines where the bug could originate. 
        Do not change the overall logic and structure of the code.
        Return the complete, corrected code between the following pattern: start_code and end_code.
        Based on the bug description follow the following guidelines for repair: 
        1. Conditional Statements:
        If there's a defect in conditional statements, invert the condition of the code block.
        2. Sensitivity List:
        If the issue lies in the sensitivity list, verify that the always block triggers appropriately. This includes triggering the always block on the following conditions:
        Signal's falling edge or Signal's rising edge or any change to a variable within the block or Signal's level change.
        3. Assignment Block:
        When dealing with assignment block defects:
        Change a blocking assignment to nonblocking or Convert a non-blocking assignment to blocking.
        4. Numeric Value:
        For numeric value discrepancies:
        Adjust the value of an identifier by either incrementing or decrementing it by 1



        Bug Description: 
        ----------------

        '{bug_description}' 


        Please find the provided buggy Verilog code below: 
        ---------------- 

        {buggy_verilog_code} 

        """
    
    prompt5= f"""Think step by step. Focus on making one or more simple changes within the code, targeting the lines where the bug could originate. Provide reasoning for every code statement you generate. Ensure not to change the overall logic and structure of the code.
                 Provide the complete functioning code without adding extra words, comments, or explanations to your answer. Given the bug description: <<<{bug_description}>>>, fix the following buggy Verilog code:

                <<<{buggy_verilog_code}>>>

                Follow these repair guidelines based on the bug description: If the issue is with conditional statements, try flipping the condition. If it's about sensitivity, make sure to set triggers for when the signal goes up or down, when any variable changes, or when the signal's level changes. For assignment problems, switch between blocking and non-blocking assignments. And if there's a number problem, adjust the value by adding or subtracting 1.
                Lastly, ensure the corrected code is returned between the following pattern: start_code and end_code, ensuring there are no additional words, comments, or explanations.    """

    print("prompt =")
    print(prompt1)
    
    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt1)) 
    #return prompt
    return prompt1




def feedback_path_creation(file_name,directory_path,iteration):

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name
    

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    directory_path=directory_path+"/Run"+"_"+str(iteration)
    #directory_path = os.path.join(directory_path, "Run")
    #directory_path = os.path.join(directory_path,iteration)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    
    return directory_path
    
        




def feedback_scenario(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description,iteration,feedback_logic,current_trial):
    

    optparser = OptionParser()
    optparser.add_option("-v","--version",action="store_true",dest="showversion",
                         default=False,help="Show the version")
    optparser.add_option("-I","--include",dest="include",action="append",
                         default=[],help="Include path")
    optparser.add_option("-D",dest="define",action="append",
                         default=[],help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = [buggy_src_file_path, test_bench_path]
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")

    for f in filelist:
        if not os.path.exists(f): raise IOError("file not found: " + f)

    codegen = ASTCodeGenerator()
    print("Reacheeeeeeeeeeeeeeeeeeeeeeeeeeeeeed")
    # parse the files (in filelist) to ASTs (PyVerilog ast)
    ast, directives = parse([buggy_src_file_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    
    #ast, _ = parse([buggy_src_file_path])

    ast.show()
    print(ast)
    src_code = codegen.visit(ast)
    print(src_code)

    print("\n\n")
    
    print("file_name: ")
    print(file_name)
    print("buggy_src_file_path: ")
    print(buggy_src_file_path)
    print("directory_path: ")
    print(directory_path)
    print("PROJ_DIR: ")
    print(PROJ_DIR)

    
    if(feedback_logic!=0 and current_trial!=0):
        
        try:
            # Copy the file from source path to destination path
            dst_path = os.path.join(PROJ_DIR, file_name)
            shutil.copy(buggy_src_file_path, dst_path)
            print(f"File copied from '{buggy_src_file_path}' to '{dst_path}'.")
            print("dst_path: ")
            print(dst_path)
            print("buggy_src_file_path: ")
            print(buggy_src_file_path)
            print("directory_path: ")
            print(directory_path)
            print("PROJ_DIR: ")
            print(PROJ_DIR)
            

            # Delete the copied file
        except FileNotFoundError:
            print("File not found.")
        except Exception as e:
            print(f"An error occurred: {e}")



    mutation_op = MutationOp(0, True, True)
    orig_fitness, sim_time = calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path)
    print("orig_fitness = ")
    print(orig_fitness)
    
    mismatch_set, uniq_headers = get_output_mismatch(TB_ID,oracle_path)
    print(mismatch_set)
    
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

    comp_failures = 0
    
    if mutation_op.fault_loc:
        tmp_mismatch_set = copy.deepcopy(mismatch_set)
        print()
        mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers) # compute fault localization for the parent
        print("Initial Fault Localization:", str(mutation_op.fault_loc_set))
        while len(mutation_op.new_vars_in_fault_loc) > 0:
            new_mismatch_set = set(mutation_op.new_vars_in_fault_loc.values())
            print("New vars in fault loc:", new_mismatch_set)
            mutation_op.new_vars_in_fault_loc = dict()
            tmp_mismatch_set = tmp_mismatch_set.union(new_mismatch_set)
            mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers)
            print("Fault Localization:", str(mutation_op.fault_loc_set))
        print("Final mismatch set:", tmp_mismatch_set)
        print("Final Fault Localization:", str(mutation_op.fault_loc_set))
        print(len(mutation_op.fault_loc_set))
        # print(mutation_op.stoplist)
        # print(mutation_op.wires_brought_in)
        
                # exit(1)

        mutation_op.implicated_lines = set()
        mutation_op.collect_lines_for_fl(ast)
        print("Lines implicated by FL: %s" % str(mutation_op.implicated_lines))
        print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_lines))

    if(feedback_logic!=0 and current_trial!=0):
        try:
            os.remove(dst_path)
            print(f"File deleted from '{dst_path}'.")
        except FileNotFoundError:
            print("File not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

   
    f_name=file_name.split('.v')[0]
    '''
    directory_path=directory_path+f_name
    

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    '''
    directory_path=directory_path+"/iteration"+"_"+str(current_trial)


    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    #qqqqqqqqqqqqq
    basic_path = f_name+'_prompt_feedback'
    output_file_path = os.path.join(directory_path, basic_path)
    
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()
    
    
    prompt4= f""" 

        Your Task: 
        ----------------
        Correct a buggy Verilog code provided below. You'll be given: 

        1. The buggy Verilog code. 
        2. A description of the bug. 
        3. A list of potential elements where the bug might originate. 

        Instructions: 
        ----------------

        Deliver a complete, functional code without additional words, comments, or explanations. 
        Focus on making one or more simple changes within the code, targeting the lines where the bug could originate. 
        Do not change the overall logic and structure of the code. 
        After generating the code, append a distinct section at the end, labeled "Code Changes". In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.


        Bug Description: 
        ----------------

        '{bug_description}' 

        Identified Differences: 
        ---------------- 

        After comparing the output of the correct and buggy code, we have identified elements that may be contributing to the bug. These elements, including Input Ports, Output Ports, Registers, Wires, or Data Types, will be listed below: 

        {tmp_mismatch_set} 
        

        Please find the provided buggy Verilog code below: 
        ---------------- 

        {buggy_verilog_code} 

        """ 
    

    print("prompt =")
    print(prompt4)
    
    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt4)) 
    
    #return prompt
    return prompt4 ,orig_fitness 

def get_first_part(content):
    # Split the content into two parts based on "Code Changes"
    parts = content.split("Code Changes", 1)
    
    # Check if there are two parts
    if len(parts) == 2:
        # Get the first part before "Code Changes"
        first_part = parts[0].strip()
    else:
        # If "Code Changes" is not found, set first_part to None
        first_part = None
    
    return first_part

def get_second_part(content):
    # Split the content into two parts based on "Code Changes"
    parts = content.split("Code Changes", 1)
    
    # Check if there are two parts
    if len(parts) == 2:
        # Get the second part after "Code Changes", stripping any leading and trailing whitespace and comments
        second_part = parts[1].strip().lstrip('/*').rstrip('*/').strip()
    else:
        # If "Code Changes" is not found, set second_part to None
        second_part = None
    
    return second_part




def split_string(content):
    # Split the content into two parts based on "Code Difference"
    parts = content.split("Code Changes", 1)

    # Check if there are two parts
    if len(parts) == 2:
        # Get the second part after "Code Difference"
        second_part = "Code Changes" + parts[1].strip()  # Add "Code Changes" to the beginning of the second part
    else:
        second_part = None  # If "Code Changes" is not found, set second_part to None

    return second_part

def get_completion_from_messages(messages):
    api_key = OPENAI_API_KEY
    #client = OpenAI(api_key=api_key)
    client = OpenAI(organization= organization_key,api_key=api_key)
    #client.models.list()
    #model_type = "gpt-3.5-turbo-1106"
    model_type="gpt-4"

    '''
    for item in messages[1:]:
        print(item['content'])
        num_tokens_prompt_before=num_tokens_from_string(item['content'], model_type)
    '''

    num_tokens_prompt_before = 0  # Initialize the variable to accumulate the number of tokens

    # Iterate over the list starting from the second element
    for item in messages[1:]:
        num_tokens_prompt_before += num_tokens_from_string(item['content'], model_type)
        print(item['content'])

    print("\n aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \n")
    print(messages)
    print("\n aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \n")
    print("num_tokens_prompt_before = ")
    print(num_tokens_prompt_before)
    #############################
    ##start_time
    t_start = time.time()


    response = client.chat.completions.create(
        model=model_type,
        messages=messages
         # this is the degree of randomness of the model's output
    )
    t_finish = time.time()
    
    total_time= t_finish - t_start
    ## finish_time
    ###################################
    
    num_tokens_output_after=num_tokens_from_string(str(response.choices[0].message.content), model_type)
    
    ##cost_calculation
    if model_type=="gpt-3.5-turbo-1106":
        cost_before= ((num_tokens_prompt_before)*(0.0010))/(1000)
        cost_after= ((num_tokens_output_after)*(0.0020))/(1000)
        total_cost = cost_before + cost_after
        
    if model_type=="gpt-4":
        cost_before= ((num_tokens_prompt_before)*(0.03))/(1000)
        cost_after= ((num_tokens_output_after)*(0.06))/(1000)
        total_cost = cost_before + cost_after

#     print(str(response.choices[0].message))
    #return str(response.choices[0].message.content),model_type
    return str(response.choices[0].message.content),num_tokens_prompt_before,num_tokens_output_after,cost_before,cost_after,total_cost,model_type,total_time

def collect_messages1(prompt):
    
    print("context=")
    print(context)
    context.append({'role':'user', 'content':f"{prompt}"})
    print("context=")
    print(context)
    print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    response,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time = get_completion_from_messages(context) 
    print(response)
    second_part = split_string(response)
    if second_part:
        print("Second Part (after 'Code Difference'):")
        print(second_part)
    else:
        print("No 'Code Difference' found.")

    #test_part
    #context.append({'role':'assistant', 'content':f"{response}"})
    #print("context=")
    #print(context)
    
    
    return response , second_part ,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time

def collect_messages2(prompt):
    
    print("context=")
    print(context)
    context.append({'role':'user', 'content':f"{prompt}"})
    print("context=")
    print(context)
    response,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time = get_completion_from_messages(context) 
    print(response)
    second_part = split_string(response)
    if second_part:
        print("Second Part (after 'Code Difference'):")
        print(second_part)
    else:
        print("No 'Code Difference' found.")

    #test_part
    #context.append({'role':'assistant', 'content':f"{response}"})
    #print("context=")
    #print(context)
    
    
    return response , second_part ,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time

def collect_messages3(prompt):
    
    print("context=")
    print(context)
    #context.append({'role':'user', 'content':f"{prompt}"})
    #print("context=")
    #print(context)
    response,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time = get_completion_from_messages(context) 
    print(response)
    second_part = split_string(response)
    if second_part:
        print("Second Part (after 'Code Difference'):")
        print(second_part)
    else:
        print("No 'Code Difference' found.")

    #test_part
    #context.append({'role':'assistant', 'content':f"{response}"})
    #print("context=")
    #print(context)
    
    
    return response , second_part ,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time

def collect_messages4(prompt,path_select):
    
    #print("context=")
    #print(context)
    if path_select == 1:
        context.append({'role':'user', 'content':f"{prompt}"})
        print("select path ==1")


    #print("context=")
    #print(context)
    response,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time = get_completion_from_messages(context) 
    #print(response)
    second_part = split_string(response)
    if second_part:
        print("Second Part (after 'Code Difference'):")
        print(second_part)
    else:
        print("No 'Code Difference' found.")

    #test_part
    #context.append({'role':'assistant', 'content':f"{response}"})
    #print("context=")
    #print(context)
    
    
    return response , second_part ,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time    

def send_prompt_chatgpt(prompt):
    #evaluate number of tokens input and output chatgpt use chatgpt tokenizer
    #cost
    #evaluate time 
    
    api_key = OPENAI_API_KEY
    #client = OpenAI(api_key=api_key)
    client = OpenAI(organization=organization_key,api_key=api_key)
    #client.models.list()
    #model_type = "gpt-3.5-turbo-1106"
    model_type="gpt-4"
    #model_type="gpt-4o"
    num_tokens_prompt_before=num_tokens_from_string(prompt, model_type)
    #num_tokens_prompt_before=num_tokens_from_string_gpto(prompt, model_type)
    
    #num_tokens_prompt_before=0
    #############################
    ##start_time
    t_start = time.time()
    
    completion = client.chat.completions.create(
    #model="gpt-4",
    model=model_type,
    messages=[
        {"role": "system", "content": "You are a helpful assistant for fixing Verilog and system Verilog code."},
        {"role": "user", "content": prompt}
    ]
     
    )
    t_finish = time.time()
    
    total_time= t_finish - t_start
    ## finish_time
    ###################################
    
    num_tokens_output_after=num_tokens_from_string(str(completion.choices[0].message.content), model_type)
    #num_tokens_output_after=num_tokens_from_string_gpto(str(completion.choices[0].message.content), model_type)
    #num_tokens_output_after=0
    
    ##cost_calculation
    if model_type=="gpt-3.5-turbo-1106":
        cost_before= ((num_tokens_prompt_before)*(0.0010))/(1000)
        cost_after= ((num_tokens_output_after)*(0.0020))/(1000)
        total_cost = cost_before + cost_after
        
    if model_type=="gpt-4":
        cost_before= ((num_tokens_prompt_before)*(0.03))/(1000)
        cost_after= ((num_tokens_output_after)*(0.06))/(1000)
        total_cost = cost_before + cost_after

    if model_type=="gpt-4o":
        cost_before= ((num_tokens_prompt_before)*(0.0005))/(1000)
        cost_after= ((num_tokens_output_after)*(0.0010))/(1000)
        total_cost = cost_before + cost_after
    
    return str(completion.choices[0].message.content),num_tokens_prompt_before,num_tokens_output_after,cost_before,cost_after,total_cost,model_type,total_time


def techniques_main_output(output_postprocess,iteration,scenario_ID,file_name,model_type,directory_path):
    

    f_name=file_name.split('.v')[0]
    
    directory_output= directory_path +f_name+"/techniques_main_output"
       
        
    if not os.path.exists(directory_output):
        os.makedirs(directory_output)
        print(f"Directory '{directory_output}' created.")
    else:
        print(f"Directory '{directory_output}' already exists.")
        
        
        
    init_output_file_name_before = f_name+'_iter_'+str(iteration)+'_Sc_ID_'+str(scenario_ID)+"_model_type_"+str(model_type)+'.txt'

    
    output_file_path_before = os.path.join(directory_output, init_output_file_name_before)
    
    
    with open(output_file_path_before, 'w') as file_b:
        file_b.write(str(output_postprocess))
    
        



def gpt_output_postprocessing(output_postprocess,iteration,scenario_ID,file_name,model_type,directory_path):
    

    f_name=file_name.split('.v')[0]
    
    directory_preprocessing= directory_path +f_name+"/output_preprocessing"
    directory_postprocessing= directory_path +f_name+"/output_postprocessing"
       
        
    if not os.path.exists(directory_preprocessing):
        os.makedirs(directory_preprocessing)
        print(f"Directory '{directory_preprocessing}' created.")
    else:
        print(f"Directory '{directory_preprocessing}' already exists.")
        
    if not os.path.exists(directory_postprocessing):
        os.makedirs(directory_postprocessing)
        print(f"Directory '{directory_postprocessing}' created.")
    else:
        print(f"Directory '{directory_postprocessing}' already exists.")
        
        
        
    init_output_file_name_before = f_name+'_iter_'+str(iteration)+'_Sc_ID_'+str(scenario_ID)+"_model_type_"+str(model_type)+'.v'
    init_output_file_name_after = f_name+'_iter_'+str(iteration)+'_Sc_ID_'+str(scenario_ID)+"_model_type_"+str(model_type)+'.v'
    
    output_file_path_before = os.path.join(directory_preprocessing, init_output_file_name_before)
    output_file_path_after = os.path.join(directory_postprocessing, init_output_file_name_after)
    
    
    with open(output_file_path_before, 'w') as file_b:
        file_b.write(str(output_postprocess))
    
    
 
    new_string = output_postprocess.replace("```verilog", "")
    new_string = new_string.replace("```", "")
    
    with open(output_file_path_after, 'w') as file_f:
        file_f.write(str(new_string))
        
    return init_output_file_name_after, output_file_path_after ,directory_postprocessing 


def gpt_output_postprocessing_feedback(output_postprocess,current_trial,scenario_ID,file_name,model_type,directory_path,main_prompt):
    

    f_name=file_name.split('.v')[0]
    
    directory_preprocessing= directory_path+"/iteration"+"_"+str(current_trial)+"/output_preprocessing"
    directory_postprocessing= directory_path+"/iteration"+"_"+str(current_trial)+"/output_postprocessing"
    #directory_main_prompt= directory_path+"/iteration"+"_"+str(current_trial)+"/main_prompt_send"
       
        
    if not os.path.exists(directory_preprocessing):
        os.makedirs(directory_preprocessing)
        print(f"Directory '{directory_preprocessing}' created.")
    else:
        print(f"Directory '{directory_preprocessing}' already exists.")
        
    if not os.path.exists(directory_postprocessing):
        os.makedirs(directory_postprocessing)
        print(f"Directory '{directory_postprocessing}' created.")
    else:
        print(f"Directory '{directory_postprocessing}' already exists.")

    '''
    if not os.path.exists(directory_main_prompt):
        os.makedirs(directory_main_prompt)
        print(f"Directory '{directory_main_prompt}' created.")
    else:
        print(f"Directory '{directory_main_prompt}' already exists.")
    '''   
        
        
        
    #init_output_file_name_before = f_name+'_iter_'+str(current_trial)+'_Sc_ID_'+str(scenario_ID)+"_model_type_"+str(model_type)+'.v'
    #init_output_file_name_after = f_name+'_iter_'+str(current_trial)+'_Sc_ID_'+str(scenario_ID)+"_model_type_"+str(model_type)+'.v'
    init_output_file_name_before = f_name+'_iter_'+str(current_trial)+'.v'
    init_output_file_name_after = f_name+'_iter_'+str(current_trial)+'.v'
    #main_prompt_output = f_name+'_iter_'+str(current_trial)+'_Sc_ID_'+str(scenario_ID)+"_model_type_"+str(model_type)+'.txt'
    
    output_file_path_before = os.path.join(directory_preprocessing, init_output_file_name_before)
    output_file_path_after = os.path.join(directory_postprocessing, init_output_file_name_after)
    #output_file_main_prompt_path = os.path.join(directory_main_prompt, main_prompt_output)
    
    with open(output_file_path_before, 'w') as file_b:
        file_b.write(str(output_postprocess))
    
    #with open(output_file_main_prompt_path, 'w') as file_c:
    #    file_c.write(str(main_prompt))
    
 
    new_string = output_postprocess.replace("```verilog", "")
    new_string = new_string.replace("```", "")
    
    with open(output_file_path_after, 'w') as file_f:
        file_f.write(str(new_string))
        
    return init_output_file_name_after, output_file_path_after ,directory_postprocessing 


def save_output_from_gpt(file_name,scenario_ID,Iteration_number,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time):
    # create and save to pandas
    

    # Load CSV file into a pandas DataFrame
    pandas_save_results = pd.read_csv(pandas_save_result_path)
    new_row_values = [len(pandas_save_results),file_name,scenario_ID,Iteration_number,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time]  # Replace with your values
    #print(pandas_save_results)
    pandas_save_results.loc[len(pandas_save_results)] = new_row_values
    #print(pandas_save_results)
    # Step 3: Save the updated DataFrame to the CSV file
    pandas_save_results.to_csv(pandas_save_result_path, index=False)
    #filename
    #scenario_ID
    #iteration_number
    #cost input
    #cost output
    #cost_total
    #model type
    #fitness value
    #simulation pass or fail
    # fix or not
    #save time calculation
    #save token number before and after 

def save_feedback_output_from_gpt(file_name,scenario_ID,run_number,Iteration_number,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time):
    # create and save to pandas
    # Load CSV file into a pandas DataFrame
    pandas_save_results = pd.read_csv(pandas_save_feedback_result_path)
    new_row_values = [len(pandas_save_results),file_name,scenario_ID,run_number,Iteration_number,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time]  # Replace with your values
    #print(pandas_save_results)
    pandas_save_results.loc[len(pandas_save_results)] = new_row_values
    #print(pandas_save_results)
    # Step 3: Save the updated DataFrame to the CSV file
    pandas_save_results.to_csv(pandas_save_feedback_result_path, index=False)




def run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory):

    print("Running VCS simulation")
    #os.system("cat %s" % fileName)
    t_start = time.time()
    
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")
    # get the filename only if full path specified
    #if "/" in output_file_path_after: output_file_path_after = output_file_path_after.split("/")[-1]

    try:
        # Extract the filename from the original path
        filename = os.path.basename(output_file_path_after)
        print("output_file_path_after=")
        print(output_file_path_after)
        print("filename=")
        print(filename)
        # Create the new path by combining the new directory and the original filename
        new_path = os.path.join(PROJ_DIR, filename)

        # Copy the file from the original path to the new path
        shutil.copy(output_file_path_after, new_path)

        # Run the bash script using the copied file in the new path
        cmd = ["bash", EVAL_SCRIPT, orig_file_name, filename, PROJ_DIR]
        process = subprocess.Popen(cmd)
        process.wait() 

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Delete the copied file after running the script
        try:
        
            os.remove(new_path)
            print(f"Successfully deleted the copied file: {new_path}")
        except Exception as e:
            print(f"Error deleting the copied file: {e}")
    # TODO: The test bench is currently hard coded in eval_script. Do we want to change that?
    #for_testing
    #os.system("bash %s %s %s %s" % (EVAL_SCRIPT, ORIG_FILE, fileName, PROJ_DIR))
    
    #might be an answer # check it first 
    
    # Construct the new file name and path
    new_file_name = f"{output_file_name_after}_{TB_ID}_output.txt"
    new_file_path = os.path.join(output_file_path_after_directory, new_file_name)


    #t_start = time.time()
    #cmd = ["bash", EVAL_SCRIPT, orig_file_name, output_file_path_after, PROJ_DIR]
    #process = subprocess.Popen(cmd)
    #process.wait() 
    
    
    if not os.path.exists("output_%s.txt" % TB_ID): 
        t_finish = time.time()
        return 0,False, t_finish - t_start # if the code does not compile, return 0
        # return math.inf

    f = open(oracle_path, "r")
    oracle_lines = f.readlines()
    f.close()
    
    # Rename and move the output file
    # de 7eta zyada
    #me7tag a test bel fitness 1 we m7tag an2el el file lel path el sa7
    #os.rename("output_%s.txt" % TB_ID, new_file_path)

    f = open("output_%s.txt" % TB_ID, "r")
    sim_lines = f.readlines()
    f.close()
    
        # Get the current working directory (where the Python script is located)
    current_directory = os.getcwd()

    # Specify the filename of the file you want to move
    file_to_move = "output_%s.txt" % TB_ID
    #new_file_name = f"{file_to_move}_iter{iteration}"

    # Specify the destination path
    destination_path = output_file_path_after_directory+"/output_simulation"

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        print(f"Directory '{destination_path}' created.")
    else:
        print(f"Directory '{destination_path}' already exists.")

    try:
        # Construct the full path to the file
        original_file_path = os.path.join(current_directory, file_to_move)
        # Extract the filename without the path and extension
        original_filename, original_extension = os.path.splitext(os.path.basename(original_file_path))

    # Add the iteration before the extension
        new_filename = f"{original_filename}_iter{iteration}{original_extension}"

        # Move the file to the specified destination path
        shutil.move(original_file_path, os.path.join(destination_path, new_filename))
        print(f"Successfully moved the file to: {destination_path}")

    except Exception as e:
        print(f"An error occurred: {e}")





    # 2amove el file aw a write fe path el ana 3ayzo we a3melo remove 
    # aw momken a remove delwa2ty el output 
    
    ff, total_possible = fitness.calculate_fitness(oracle_lines, sim_lines, None, "")
        
    normalized_ff = ff/total_possible
    if normalized_ff < 0: normalized_ff = 0
    print("FITNESS = %f" % normalized_ff)
    t_finish = time.time()
    # if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID) # Do we need to do this here? Does it make a difference?
    #t_finish = time.time()

        


   

     
    fitness_value=normalized_ff
    simulation_status=True
    simulation_time = t_finish - t_start
    fix_status=0
    #return normalized_ff, t_finish - t_start
    return fitness_value,simulation_status,simulation_time
    




def run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_code,file_name):

    print("Running VCS simulation")
    #os.system("cat %s" % fileName)
    t_start = time.time()
    
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")
    file_name_without_extension = os.path.splitext(file_name)[0]
    # get the filename only if full path specified
    #if "/" in output_file_path_after: output_file_path_after = output_file_path_after.split("/")[-1]

    try:
        # Extract the filename from the original path
        filename = os.path.basename(buggy_code)

        # Create the new path by combining the new directory and the original filename
        #new_path = os.path.join(PROJ_DIR, filename)

        # Copy the file from the original path to the new path
        #shutil.copy(output_file_path_after, new_path)

        # Run the bash script using the copied file in the new path
        cmd = ["bash", EVAL_SCRIPT, orig_file_name, filename, PROJ_DIR]
        #cmd = ["bash", EVAL_SCRIPT, orig_file_name, orig_file_name, PROJ_DIR]
        process = subprocess.Popen(cmd)
        process.wait() 

    except Exception as e:
        print(f"An error occurred: {e}")

    
 
    
    if not os.path.exists("output_%s.txt" % TB_ID): 
        t_finish = time.time()
        return 0,False, t_finish - t_start # if the code does not compile, return 0
        # return math.inf

    f = open(oracle_path, "r")
    oracle_lines = f.readlines()
    f.close()
    
    # Rename and move the output file
    # de 7eta zyada
    #me7tag a test bel fitness 1 we m7tag an2el el file lel path el sa7
    #os.rename("output_%s.txt" % TB_ID, new_file_path)

    f = open("output_%s.txt" % TB_ID, "r")
    sim_lines = f.readlines()
    f.close()
    
        # Get the current working directory (where the Python script is located)
    current_directory = os.getcwd()

    # Specify the filename of the file you want to move
    file_to_move = "output_%s.txt" % TB_ID
    #new_file_name = f"{file_to_move}_iter{iteration}"

    # Specify the destination path
    destination_path = current_directory+"/baseline_output_simulation_correct"

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        print(f"Directory '{destination_path}' created.")
    else:
        print(f"Directory '{destination_path}' already exists.")

    try:
        # Construct the full path to the file
        original_file_path = os.path.join(current_directory, file_to_move)
        # Extract the filename without the path and extension
        original_filename, original_extension = os.path.splitext(os.path.basename(original_file_path))

    # Add the iteration before the extension
        new_filename = f"{file_name_without_extension}{original_extension}"
        #new_filename = f"{original_filename}{original_extension}"

        # Move the file to the specified destination path
        shutil.move(original_file_path, os.path.join(destination_path, new_filename))
        print(f"Successfully moved the file to: {destination_path}")

    except Exception as e:
        print(f"An error occurred: {e}")





    # 2amove el file aw a write fe path el ana 3ayzo we a3melo remove 
    # aw momken a remove delwa2ty el output 
    
    ff, total_possible = fitness.calculate_fitness(oracle_lines, sim_lines, None, "")
        
    normalized_ff = ff/total_possible
    if normalized_ff < 0: normalized_ff = 0
    print("FITNESS = %f" % normalized_ff)
    t_finish = time.time()
    # if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID) # Do we need to do this here? Does it make a difference?
    #t_finish = time.time()

        


   

     
    fitness_value=normalized_ff
    simulation_status=True
    simulation_time = t_finish - t_start
    fix_status=0
    #return normalized_ff, t_finish - t_start
    return fitness_value,simulation_status,simulation_time
    
def save_baseline_output(file_name,fitness_value,simulation_status,simulation_time):
    # create and save to pandas
    

    # Load CSV file into a pandas DataFrame
    pandas_save_results = pd.read_csv(baseline_save_results)
    new_row_values = [len(pandas_save_results),file_name,fitness_value,simulation_status,simulation_time]  # Replace with your values
    #print(pandas_save_results)
    pandas_save_results.loc[len(pandas_save_results)] = new_row_values
    #print(pandas_save_results)
    # Step 3: Save the updated DataFrame to the CSV file
    pandas_save_results.to_csv(baseline_save_results, index=False)
    #filename
    #scenario_ID
    #iteration_number
    #cost input
    #cost output
    #cost_total
    #model type
    #fitness value
    #simulation pass or fail
    # fix or not
    #save time calculation
    #save token number before and after 





def main(args):
    global context 
    file_path = args.pandas_csv_path
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    directory_path=directory_creation(args.scenario_ID,args.experiment_number)
    start_index = 0


    if (args.scenario_ID == 0):#0= basic_scenario
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #result = df[df['file_name'] == file_name]
            #print(result)
            #prompt=basic_scenario(file_name,buggy_src_file_path,directory_path)
            
            #output_postprocess=send_prompt_chatgpt(prompt)
            #gpt_output_postprocessing(output_postprocess)
        elif(args.choose_file == "all") :
            #loop over pandas dataframe 
            #for each file in pandas dataframe column call function basicscenario
            # and for each file with this prompt run n number of iterations 
            # for each iteration send the output to chatgpt 
            # then process the output data 
            # then save the output data in a file 
            #for index, row in df.iterrows():
             # Iterate over rows starting from the specified index
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = basic_scenario(file_name,buggy_src_file_path,directory_path)
                for iteration in range(1, iterations + 1):
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    

                    # I added this to check if the simulation is working correctly as I added the needed files and copied the correct code to the output from chatgpt file to make sure that I will get fitness =1
                    #output_file_name_after = "decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after = "/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing/decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after_directory = "/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing"
                    
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                    
                    #for getting baseline fitness
                    #save_baseline_output(file_name,fitness_value,simulation_status,simulation_time)
                    #print(output_postprocess)
                    print("/////////////////////////////////////////////////////////")
                    #gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                #aaaaaaaaa
                    #gpt_output_postprocessing(file_name,iteration)
        else:
            print("wrong value")

    elif (args.scenario_ID == 1):#1= bug_description_scenario
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #result = df[df['file_name'] == file_name]
            #print(result)
            #prompt=basic_scenario(file_name,buggy_src_file_path,directory_path)
            
            #output_postprocess=send_prompt_chatgpt(prompt)
            #gpt_output_postprocessing(output_postprocess)
        elif(args.choose_file == "all") :
            #loop over pandas dataframe 
            #for each file in pandas dataframe column call function basicscenario
            # and for each file with this prompt run n number of iterations 
            # for each iteration send the output to chatgpt 
            # then process the output data 
            # then save the output data in a file 
            #for index, row in df.iterrows():
             # Iterate over rows starting from the specified index
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = bug_description_scenario(file_name,buggy_src_file_path,directory_path,bug_description)

                for iteration in range(1, iterations + 1):
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    

                    # I added this to check if the simulation is working correctly as I added the needed files and copied the correct code to the output from chatgpt file to make sure that I will get fitness =1
                    #output_file_name_after = "decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after = "/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing/decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after_directory = "/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing"
                    
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                    
                    #for getting baseline fitness
                    #save_baseline_output(file_name,fitness_value,simulation_status,simulation_time)
                    #print(output_postprocess)
                    print("/////////////////////////////////////////////////////////")
                    #gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                #aaaaaaaaa
                    #gpt_output_postprocessing(file_name,iteration)
        else:
            print("wrong value")   


    elif (args.scenario_ID == 2):#2= scenario_mismatch
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #result = df[df['file_name'] == file_name]
            #print(result)
            #prompt=basic_scenario(file_name,buggy_src_file_path,directory_path)
            
            #output_postprocess=send_prompt_chatgpt(prompt)
            #gpt_output_postprocessing(output_postprocess)
        elif(args.choose_file == "all") :
            #loop over pandas dataframe 
            #for each file in pandas dataframe column call function basicscenario
            # and for each file with this prompt run n number of iterations 
            # for each iteration send the output to chatgpt 
            # then process the output data 
            # then save the output data in a file 
            #for index, row in df.iterrows():
             # Iterate over rows starting from the specified index
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = scenario_mismatch(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description)
               
                for iteration in range(1, iterations + 1):
                    
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    

                    # I added this to check if the simulation is working correctly as I added the needed files and copied the correct code to the output from chatgpt file to make sure that I will get fitness =1
                    #output_file_name_after = "decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after = "/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing/decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after_directory = "/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing"
                    
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                    
                    print("/////////////////////////////////////////////////////////")

    elif (args.scenario_ID == 3):#3= new technique adding extra part to the prompt -->lets think step by step. give reasoning for every code statement you generate and then finally write the complete generated code 
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #result = df[df['file_name'] == file_name]
            #print(result)
            #prompt=basic_scenario(file_name,buggy_src_file_path,directory_path)
            
            #output_postprocess=send_prompt_chatgpt(prompt)
            #gpt_output_postprocessing(output_postprocess)
        elif(args.choose_file == "all") :
            #loop over pandas dataframe 
            #for each file in pandas dataframe column call function basicscenario
            # and for each file with this prompt run n number of iterations 
            # for each iteration send the output to chatgpt 
            # then process the output data 
            # then save the output data in a file 
            #for index, row in df.iterrows():
             # Iterate over rows starting from the specified index
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = scenario_tech1(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description)
               
                for iteration in range(1, iterations + 1):
                    
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                    
                    print("output_postprocess=")
                    print(output_postprocess)

                    techniques_main_output(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    #parsed_output= extract_fixed_code_from_json(output_postprocess)
                    #parsed_output=output_postprocess[output_postprocess.find("{"): output_postprocess.find("}")+1]

                    try:
                        parsed_output = output_postprocess.split("start_code")[1].split("end_code")[0]
                        print("parsed_output=")
                        print(parsed_output)
                        output_postprocess= parsed_output
                        #parsed1 = json.loads(parsed_output)
                        #code_fixed = parsed1['fixed_code']
                        #code_fixed = parsed_output
                    except Exception as e:
                        print(f"An error occurred: {e}")


                    #parsed_output= output_postprocess.split("start_code")[1].split("end_code")[0]
                    #print("parsed_output=")
                    #print(parsed_output)
                    #parsed1=json.loads(parsed_output)
                    #code_fixed=parsed1['fixed_code']
                    #code_fixed=parsed_output
                    print("output_postprocess=")
                    print(output_postprocess)
                    #aaaaaaaaaaaa
                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    

                    # I added this to check if the simulation is working correctly as I added the needed files and copied the correct code to the output from chatgpt file to make sure that I will get fitness =1
                    #output_file_name_after = "decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after = "/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing/decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after_directory = "/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing"
                    
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                    
                    print("/////////////////////////////////////////////////////////")
                    

    elif (args.scenario_ID == 4):#4= new technique adding extra part to the prompt -->Start by listing the lines of code from the provided mismatch list that might be causing errors. Then, analyze which of these lines need to be corrected. Finally, generate the complete corrected code incorporating the necessary fixes. 
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #result = df[df['file_name'] == file_name]
            #print(result)
            #prompt=basic_scenario(file_name,buggy_src_file_path,directory_path)
            
            #output_postprocess=send_prompt_chatgpt(prompt)
            #gpt_output_postprocessing(output_postprocess)
        elif(args.choose_file == "all") :
            #loop over pandas dataframe 
            #for each file in pandas dataframe column call function basicscenario
            # and for each file with this prompt run n number of iterations 
            # for each iteration send the output to chatgpt 
            # then process the output data 
            # then save the output data in a file 
            #for index, row in df.iterrows():
             # Iterate over rows starting from the specified index
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = scenario_tech2(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description)
               
                for iteration in range(1, iterations + 1):
                    
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                    
                    print("output_postprocess=")
                    print(output_postprocess)

                    techniques_main_output(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    #parsed_output= extract_fixed_code_from_json(output_postprocess)
                    #parsed_output=output_postprocess[output_postprocess.find("{"): output_postprocess.find("}")+1]

                    try:
                        parsed_output = output_postprocess.split("start_code")[1].split("end_code")[0]
                        print("parsed_output=")
                        print(parsed_output)
                        output_postprocess= parsed_output
                        #parsed1 = json.loads(parsed_output)
                        #code_fixed = parsed1['fixed_code']
                        #code_fixed = parsed_output
                    except Exception as e:
                        print(f"An error occurred: {e}")

                    #parsed_output= output_postprocess.split("start_code")[1].split("end_code")[0]
                    #print("parsed_output=")
                    #print(parsed_output)
                    #parsed1=json.loads(parsed_output)
                    #code_fixed=parsed1['fixed_code']
                    #code_fixed=parsed_output

                    
                    print("output_postprocess=")
                    print(output_postprocess)
                    #aaaaaaaaaaaa
                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    

                    # I added this to check if the simulation is working correctly as I added the needed files and copied the correct code to the output from chatgpt file to make sure that I will get fitness =1
                    #output_file_name_after = "decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after = "/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing/decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after_directory = "/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing"
                    
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                    
                    print("/////////////////////////////////////////////////////////")


    elif (args.scenario_ID == 5):#5= new technique adding extra part to the prompt --> Follow the following guidelines for repair that are mentioned in cirfix
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #result = df[df['file_name'] == file_name]
            #print(result)
            #prompt=basic_scenario(file_name,buggy_src_file_path,directory_path)
            
            #output_postprocess=send_prompt_chatgpt(prompt)
            #gpt_output_postprocessing(output_postprocess)
        elif(args.choose_file == "all") :
            #loop over pandas dataframe 
            #for each file in pandas dataframe column call function basicscenario
            # and for each file with this prompt run n number of iterations 
            # for each iteration send the output to chatgpt 
            # then process the output data 
            # then save the output data in a file 
            #for index, row in df.iterrows():
             # Iterate over rows starting from the specified index
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = scenario_tech3(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description)
               
                for iteration in range(1, iterations + 1):
                    
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                    
                    print("output_postprocess=")
                    print(output_postprocess)

                    techniques_main_output(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)

                    try:
                        parsed_output = output_postprocess.split("start_code")[1].split("end_code")[0]
                        print("parsed_output=")
                        print(parsed_output)
                        output_postprocess= parsed_output
                        #parsed1 = json.loads(parsed_output)
                        #code_fixed = parsed1['fixed_code']
                        #code_fixed = parsed_output
                    except Exception as e:
                        print(f"An error occurred: {e}")
                    
                    #parsed_output= output_postprocess.split("start_code")[1].split("end_code")[0]
                    #print("parsed_output=")
                    #print(parsed_output)
                    

                    
                    print("output_postprocess=")
                    print(output_postprocess)
                    #aaaaaaaaaaaa
                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    

                    # I added this to check if the simulation is working correctly as I added the needed files and copied the correct code to the output from chatgpt file to make sure that I will get fitness =1
                    #output_file_name_after = "decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after = "/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing/decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after_directory = "/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing"
                    
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                    
                    print("/////////////////////////////////////////////////////////")
                   
    else:
        print("wrong value")

    # Your code here
    print("Arguments:", args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    
    # Define command-line arguments
    parser.add_argument('pandas_csv_path', type=str, help='Path to the csv file')
    parser.add_argument('number_iterations', type=int, help='Number of iterations to repeat passing the same prompt to gpt')
    parser.add_argument('choose_file', type=str, help='choose file name to test')#add specific file name or "all" to process on all files
    parser.add_argument('scenario_ID', type=int, help='chooses the prompt scneario')
    parser.add_argument('experiment_number', type=str, help='write the experiment number')
    parser.add_argument('feedback_logic', type=int, help='choose your feedback logic')
   # parser.add_argument('--optional_arg', type=float, default=1.0, help='Description of optional_arg')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)