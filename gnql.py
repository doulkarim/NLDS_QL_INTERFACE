# -*- coding: utf-8 -*-
"""gnlql.ipynb


#**Parsing according to our grammar**

To parsing the queries with our grammr

## **Vocaburay extraction**
"""

# Importation des librairies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # to do the grid of plots
import re
from nltk.probability import FreqDist
from nltk import Tree
from nltk.tree import *
from nltk.stem import *
# import deplacy
from nltk.parse.generate import generate
from spacy.matcher import Matcher
from nltk.tokenize import RegexpTokenizer
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from spacy import displacy
from nltk.collocations import *
from nltk.util import ngrams
from nltk.util import ngrams
## Graph library
import py2neo
from py2neo import Node,Relationship,Graph,Path,Subgraph
from py2neo import NodeMatcher,RelationshipMatcher

# queries speech packages

import time
import datetime
from pathlib import Path
import subprocess
import os

from nltk.parse.generate import generate

try:
    stopwords = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    stopwords = set(stopwords.words('english'))
#stopwords

nltk.download('punkt')
nltk.download('wordnet')

import spacy

# Graph data base connection
import py2neo
from py2neo import Node,Relationship,Graph,Path,Subgraph
from py2neo import NodeMatcher,RelationshipMatcher

def gram_contruct(properties):
  lemme=[]
  # To remove Whitespace 
  properties=re.sub(' +', ' ',properties)
  for lem in properties.split(" "):
    lemme.append('"{0}"'.format(lem))
  return lemme

def Vocabulary_extractor(url,username,password ):
    global graph
    global grammar

    graph = Graph(url, auth=(username, password))
    value_node=[]
    properties_key=[]
    val_node=[]
    prop_key=[]
    label_node=list(graph.schema.node_labels) # Label node extraction 
    label_relation=list(graph.schema.relationship_types) # label relation extraction   
    node_matcher = NodeMatcher(graph)
    relationship_matcher=RelationshipMatcher(graph)
    for lab_node in label_node:
      # node.append(list(node_matcher.match('{0}').first()).format(labe_node))
        prop_key.append(list(node_matcher.match(lab_node).first().keys()))
        val_node.append(list(node_matcher.match(lab_node).first().values()))

    # To remove duplicate
    for prop in prop_key:
      for lem in prop:
        if lem not in properties_key:
           properties_key.append(lem)
    for val in val_node:
      for v in val:
        if v not in value_node:
           value_node.append(v)
    
    Label_node=[]
    Label_relation=[]
    Keys=[]
    Values=[]
    for node in label_node:
      Label_node.append ('{0} |'.format(" ".join(gram_contruct(node))))
    for rel in label_relation:
      Label_relation.append('{0} |'.format(" ".join(gram_contruct(rel))))
    for properties in properties_key:
        Keys.append('{0} |'.format(" ".join(gram_contruct(properties))))
    for value in value_node:
        Values.append('{0} |'.format(" ".join(gram_contruct(value))))

    Label_Node=" ".join(Label_node)
    Label_Relation=" ".join(Label_relation)
    Properties_Key=" ".join(Keys)
    Properties_Value=" ".join(Values)

    ## grammar 

    grammar = nltk.CFG.fromstring("""

    Query -> Action Struct|Action DS|Action Struct DS|Action DS Struct|Action Struct DS Execution|Action DS Struct Execution|Action Struct Execution|Action DS Execution

    Struct -> DS_Selection |Properties DS_Selection|DS_Selection Properties|Properties DS_Selection Properties|Properties|DS_Selection Properties|Getting DS_Selection Properties|Properties Properties|Getting|Getting Properties|Getting Properties Properties|DS_Selection Properties|Getting DS_Selection Properties|Condition Getting DS Properties|Condition Getting|Condition Getting Properties|Condition Getting Properties Properties|Condition Getting DS_Selection Properties

    Properties -> Node|Relation|Node Relation|Node Node Relation|Node Node Relation Relation|Node Relation Node|Node Node Node Relation|Node Node Node Relation Relation Relation|Node Relation Relation|Node Node Relation Relation Relation

    Action -> "Get" |"get"| "Find" | "Give"|"get"| "find" | "give"|"Get" "all" | "Find" "all" | "Give" "all"|"get" "all" | "find" "all" | "give" "all"| "shortest" "path"|"give" "long" "shortest" "path"|"how" "many"|"number" "of"|"get" "max" |"get" "min"|"get" "avg"|"give" "max" |"give" "min"|"give" "avg"|"give" "count"|"give"|"create"|"estimate"|"visualize"|"create" "and" "estimate"|"drop"

    Getting -> "statistic"|"node"|"connection"|"property"|"relation"|"number"|"number" "connection" "by"|"max" "connection" "by"|"min" "connection" "by"|"schema"|"score"|"number" "connection"

    DS -> Algo Pattern|Pattern|Algo|Algo Name|Condition Algo Pattern|Condition Pattern|Condition Algo|Algo Condition Pattern|Condition Algo Condition Pattern

    DS_Selection -> Condition DS_Getting |DS_Getting Condition DS_Getting

    DS_Getting -> "most""important"|"less" "important"|"most" "popular"|"less" "popular"|"most" "influent"|"less" "influent"|"easy" "controle" |"easy" "reach"|"subgroup"|"group"|"optimal"|"maximum"|"shortest" "path"

    Execution -> Condition "max" "iterations" MaxIterations|Condition "max" "iterations" MaxIterations Condition "damping" "factor" Damping  

    Damping -> "0.20"|"0.25"|"0.30"|"0.35"|"0.40"|"0.45"|"0.50"|"0.55"|"0.60"|"0.65"|"0.70"|"0.75"|"0.80"|"0.85"|"0.90"|"0.95"|"1"

    Algo -> Algo_name|Algo_name MaxIterations|Algo_name "with" "max" "iterations" MaxIterations  

    Algo -> Algo_name|Algo_name MaxIterations|Algo_name "with" "max" "iterations" MaxIterations  

    Algo_name -> "pageRank"|"Label_propagation|"Weakly|"Connected" "Components" |"Louvain" |"Node similarity"

    MaxIterations -> "10" |"15"| "20"|"25"|"30"|"35"|"40"|"45" |"50"

    Pattern -> Pattern_Name|Pattern_Name Condition "name" Name|Pattern_Name Condition "name" Name|Pattern_Name "name" Name|Pattern_Name Condition Name|Pattern_Name Condition Name|Pattern_Name Name 

    Pattern_Name -> "subgraph" "projection" |"graph" "projection"|"subgraph"|"graph"

    Name -> "got-interactions"| "friendly"|"got"|"interactions"

    Orientation -> "orientation"|"non" "orientation"

    Node -> Lab_Node|Prop_key|Lab_Node Prop_key|Lab_Node Prop_key Condition_Union Prop_key Constraint_value|Lab_Node Prop_key Condition_Union Prop_key|Lab_Node Prop_key Constraint_value |Lab_Node Prop_key Constraint_value Condition_Union Constraint_value|Lab_Node Prop_key Constraint_value Condition_Union Constraint_value|Lab_Node Prop_key Constraint_value Condition_Union Prop_key Constraint_value |Lab_Node Prop_key Constraint_value Condition_Union Constraint_value|Lab_Node Prop_key Constraint_value Condition_Union Prop_key Constraint_value |Lab_Node Prop_key Constraint_value Condition Prop_key Constraint_value

    Relation -> Label_Relation|Label_Relation Relation_Properties_key|Label_Relation Relation_Properties_key Relation_Constraint_value

    Lab_Node -> Condition Label_Node|Label_Node|Condition Label_Node Condition Label_Node|Condition Label_Node|Condition Label_Node "node"|Label_Node "node"|Condition Label_Node "node" Condition Label_Node "node"|Condition Label_Node "node"

    Prop_key -> Properties_key|Condition Properties_key|Condition Properties_key|Condition Properties_key Condition Properties_key Condition Properties_key

    Label_Node ->  {0} "#"

    Label_Relation -> Condition Label_Rel|Label_Rel|Condition Label_Rel Condition Label_Rel|Condition Label_Rel "relationship"|Condition Label_Rel "relationship" Orientation|Label_Rel "relationship" Orientation|Condition Label_Rel "relationship" Orientation  Condition Label_Rel "relationship" Orientation

    Relation_Properties_key -> "weight"

    Label_Rel -> {1}"#"

    Properties_key -> {2}"#"

    Constraint_value -> Condition Node_Value|Node_Value|Condition Node_Value Condition Node_Value

    Node_Value -> {3}"#"

    Condition -> "="|"is"|"and"| "or"|"is" "not"|"different"|"is" "sup"|"is" "inf"| "who" "has"|"which"|"with"|"of"|"in" "the"|"on" "the"|"the"| "who" |"that"|"whose"|"which" |"where"|"has" "a"|"who" "has"|"in" "other"|"in" "which" |"in" "which" "every"| "owns"|"about"|"for"|"to"|"and"|"by"

    """.format(Label_Node,Label_Relation,Properties_Key,Properties_Value))
    return Graph(url, auth=(username, password)),label_node,label_relation,properties_key,value_node

"""#**Data science queries generation**"""

def nldsQl(query):
    Tree=[]
    parser = nltk.ChartParser(grammar)
    for tree in parser.parse(query.split()):
      Tree.append(tree)
      tree.pretty_print()

  # Definition des variables en array pour pouvoir les manipuler
    action=[]
    label_node=[]
    properties_key=[]
    constraint_value=[]
    label_relation=[]
    condition=[]
    pattern_name=[]
    algo=[]
    getting= []
    orientation=[]
    inter_union=[]
    aggregate=[]
    subgraph_name=[]
    relation_properties=[]
    maxIterations=[]
    ds_condition=[]
    ds_getting=[]
    damping=[]
    cypher=None

    q=Tree[0]

    # Getting data of query
    for sub_tree in q.subtrees():
      if sub_tree.label()=="Action":
        action.append(sub_tree.leaves())
      if sub_tree.label()=="Label_Node":
        label_node.append(sub_tree.leaves())
      if sub_tree.label()=="Properties_key": 
        properties_key.append(sub_tree.leaves())
      if sub_tree.label()=="Constraint_value": 
        constraint_value.append(sub_tree.leaves())
      if sub_tree.label()=="Label_Rel": 
        label_relation.append(sub_tree.leaves())
      if sub_tree.label()=="Condition": 
        condition.append(sub_tree.leaves())  
      if sub_tree.label()=="Aggregation": 
        aggregate.append(sub_tree.leaves()) 
      if sub_tree.label()=="Pattern_Name": 
        pattern_name.append(sub_tree.leaves())
      if sub_tree.label()=="Name": 
        subgraph_name.append(sub_tree.leaves())
      if sub_tree.label()=="Algo_name": 
        algo=sub_tree.leaves()
      if sub_tree.label()=="Getting": 
        getting.append(sub_tree.leaves())
      if sub_tree.label()=="MaxIterations": 
        maxIterations=sub_tree.leaves()
      if sub_tree.label()=="DS_Condition": 
        ds_condition=sub_tree.leaves()
      if sub_tree.label()=="DS_Getting": 
        ds_getting=sub_tree.leaves()
      if sub_tree.label()=="Relation_Properties_key": 
        relation_properties.append(sub_tree.leaves())
      if sub_tree.label()=="Orientation": 
         orientation.append(sub_tree.leaves())
      if sub_tree.label()=="Damping": 
         damping.append(sub_tree.leaves())    
    

     # Define default MaxIterations 
    if maxIterations==[]:
       maxIterations=[20]
    else:
        maxIterations=maxIterations
    # Damping definition 
    if damping==[]:
       damping=[0.85]
    else:
       damping=damping[0] # damping postion 0 extraction 

    if sub_tree.label()=="Orientation": 
         orientation.append(sub_tree.leaves())

    # contruction query with properties
    if len(label_node)>0:
        lab=[]
        for i in range(1,len(label_node)):
          lab.append(" ,'{0}' ".format(" ".join(label_node[i])))
          ## specify la mise du virgule avant ou après
        if len(subgraph_name)>0:
            lab=(" '{0}' {1} ".format(" ".join(map(str,label_node[0]))," ".join(map(str,lab))))
        else: 
          lab=(" '{0}' {1} ".format(" ".join(map(str,label_node[0]))," ".join(map(str,lab))))

            # define the properties_projection 

    if len(relation_properties)==0:
      relation_properties=[' ']
    else:
      relation_properties=relation_properties
    if len(properties_key)==0:
      properties_key=[' ']
    else:
      properties_key=properties_key
    if len(label_relation)==0:
      label_relation=['*']
    else:
      label_relation=label_relation

    if len(label_relation)>0:
        relation=[]
        for i in range(1,len(label_relation)):
            relation.append(" ,{1} ".format(" ".join(label_relation[i])))
        relation=(" {0} {1}" .format(" ".join(map(str,label_relation[0]))," ".join(map(str,relation))))
       
        ## relation properties getting 
        if len(relation_properties)>0:
              relation_key=[]
              for i in range(1,len(relation_properties)):
                  relation_key.append(" ,'{1}' ".format(" ".join(relation_properties[i])))
              relation_key=(" '{0}' {1}" .format(" ".join(map(str,relation_properties[0]))," ".join(map(str,relation_key))))
            
        else:
            relation_key=[" "]


    if len(subgraph_name)>0:
        sub_name=(" '{0}'".format(" ".join(subgraph_name[0])))
    else:
      sub_name=("'my_graph'")


    # define relation orientation
    if orientation==[["non", "orientation"]]:
      orientation=["'UNDIRECTED'"]
    else:
       orientation= ["'NATURAL'"]

      # Get statistique with node and label// Group by ( using 1 label node and one label relation )
    if getting ==[['statistic']]: 
      if len(label_node)==1:
          if len(label_relation)==1:         
                cypher=("""MATCH (n:{0})-[:{1}]->() WITH n, count(*) AS num RETURN min(num) AS min, max(num) AS max, avg(num) AS avg_interactions, stdev(num) AS stdev 
                      """.format(" ".join(map(str,label_node[0]))," ".join(map(str,label_relation[0]))))
    
    # schema Visualization 
    if action==[["visualize"]]:
      cypher= ("CALL db.schema.visualization()")
    #Drop one subgraph projection
    if action==[['drop']]:
        cypher= ("CALL gds.graph.drop({0})".format(sub_name))
      
      # Graph projection
    if action==[["create"]]:
      cypher= ("""CALL gds.graph.create({0},{{ {1}:{{properties:{{{2}}}}}}},{{ {3}:{{orientation:{4},properties:{{{5}}}}}}})
                """.format(sub_name," ".join(map(str,label_node[0]))," ".join(map(str,properties_key[0]))," ".join(map(str,label_relation[0]))," ".join(map(str,orientation)), " ".join(map(str,relation_properties))))
    if action==[['create', 'and', 'estimate']]:
        if len(algo)==0:
            cypher= ("""CALL gds.graph.create.estimate ('{0}' , '{1}') YIELD nodeCount, relationshipCount, requiredMemory
                  """.format(" ".join(map(str,label_node[0]))," ".join(map(str,label_relation[0])))) 
     
        # estimate the memory usage for graph creation and algorithm execution at the same time by using the so-called implicit graph creation     
        else:
          cypher= ("""CALL gds.{0}.stream.estimate({{nodeProjection: {1}, relationshipProjection: {2}}})
                """.format(sub_name," ".join(map(str,label_node[0]))," ".join(map(str,label_relation[0]))))

    if action==[['estimate']]:
       if len(algo) !=0:
          cypher= ("CALL gds.{0}.stream.estimate({1})".format(" ".join(map(str,algo)),sub_name)) 

    if action==[['estimate',',', 'execute','and','save']]:

       cypher= ("""CALL gds.{0}.write.estimate({1}, {{writeProperty: '{2}',maxIterations: {3},dampingFactor: {4}}}) YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory
              """.format(" ".join(map(str,algo)),sub_name," ".join(map(str,algo))," ".join(map(str,maxIterations))," ".join(map(str,damping))))
    
        # Condition for PageRank (popularity global)
    # Creation-estimation-execution           CALL gds.graph.create({0} {1}, {{ {2} : {{orientation: {3}}}}})
    if ds_getting==['most', 'important']:    
         cypher=("""CALL gds.graph.create({0}, '{1}',{{{2}: {{orientation: {3}}}}}) 
            CALL gds.pageRank.write.estimate({0}, {{writeProperty: 'pageRank',maxIterations: {4}, dampingFactor:{5}}})
            YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory
            CALL gds.pageRank.stream({0}) YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).name AS name, score
            ORDER BY score DESC LIMIT 10
            """.format(sub_name," ".join(map(str,label_node[0]))," ".join(map(str,label_relation[0]))," ".join(map(str,orientation)), " ".join(map(str,maxIterations))," ".join(map(str,damping))))

    if ds_getting==['less', 'important']:
        cypher=("""CALL gds.graph.create.estimate({0}, '{1}', {{{2}: {{orientation: {3}}}}}) 
                   CALL gds.pageRank.write({0}, {{writeProperty: 'pageRank',maxIterations: {4}, dampingFactor: {5}}})
                   YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory
                   CALL gds.pageRank.stream({0}) YIELD nodeId, score
                   RETURN gds.util.asNode(nodeId).name AS name, score
                   ORDER BY score ASC LIMIT 10
            """.format(sub_name," ".join(map(str,label_node[0]))," ".join(map(str,label_relation[0]))," ".join(map(str,orientation)), " ".join(map(str,maxIterations))," ".join(map(str,damping))))

    # Community detection
    if ds_getting==['group'] and getting==[['node']]: 

       cypher=("""CALL gds.graph.create.estimate({0}, '{1}', {{{2}: {{orientation:{3}}}}}) 
            CALL gds.pageRank.write({0}, {{writeProperty: 'pageRank',maxIterations: {4}, dampingFactor: {5}}})
            YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory
            CALL gds.pageRank.stream({0}) YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).name AS name, score
            ORDER BY score DESC LIMIT 10
            """.format(sub_name," ".join(map(str,label_node[0]))," ".join(map(str,label_relation[0]))," ".join(map(str,orientation)), " ".join(map(str,maxIterations))," ".join(map(str,damping))))
    
    # Degree centraly (Most popular about on property (ex: Popular person about followers on tweeter))

    if getting==[['number', 'connection', 'by']] or ds_getting==['most', 'popular']:
       cypher=("""MATCH (n:{0})-[r:{1}]-() 
       with n,count(*) as degree  return id(n), degree ORDER BY (degree) DESC

       """.format(" ".join(map(str,label_node[0]))," ".join(map(str,label_relation[0]))))
    
    if ds_getting==['less', 'popular']:
        cypher=("""MATCH (n:{0})-[r:{1}]-() 
        with n,count(*) as degree  return id(n), degree ORDER BY (degree) ASC

        """.format(" ".join(map(str,label_node[0]))," ".join(map(str,label_relation[0]))))

    if ds_getting==["group"] or ds_getting==["subgroup"] or ds_getting==["cluster"]:
               
        cypher=("""CALL gds.graph.create({0},{{{1}: {{properties:{2} }}}},{{{3}: {{orientation: {4},properties: {5}}}}})

              CALL gds.labelPropagation.write.estimate({0}, {{writeProperty: 'community'}})
              YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory

              CALL gds.labelPropagation.stream({0},{{maxIterations: {6}}}) 
              YIELD nodeId, communityId
              RETURN communityId, count(nodeId) AS size
              ORDER BY size DESC
              LIMIT 5


              or 

              CALL gds.louvain.write.estimate({{
                nodeProjection: '{1}',
                relationshipProjection: '{3}',
                includeIntermediateCommunities: False,
                writeProperty: "Community"}}) 
              YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory

              CALL gds.louvain.stream({{
                nodeProjection: '{1}',
                relationshipProjection: '{3}',
                includeIntermediateCommunities: true}})
              YIELD nodeId, communityId, intermediateCommunityIds
              RETURN gds.util.asNode(nodeId).id AS {0},
              communityId, intermediateCommunityIds;

              or
              
              CALL gds.wcc.stream({{nodeProjection: {1},relationshipProjection: {2}}})
              YIELD nodeId, componentId
              RETURN componentId, collect(gds.util.asNode(nodeId).id) AS {0}
              ORDER BY size({0}) DESC;

            """.format(sub_name," ".join(map(str,label_node[0]))," ".join(map(str,properties_key[0]))," ".join(map(str,label_relation[0]))," ".join(map(str,orientation))," ".join(map(str,relation_properties[0]))," ".join(map(str,maxIterations))))

     #  Pathfinding  
    if ds_getting==['shortest', 'path']:

        if len(node_value)==2: 
          if len(label_relation)==1:
             cypher=(" MATCH p = shortestPath((n1:{0} {{{1}:'{2}'}})-[r:{3}]-(n2:{4} {{{5}:'{6}'}})) return p ".format(" ".join(map(str,label_node[0]))," ".join(map(str,properties_key[0]))," ".join(map(str,node_value[0]))," ".join(map(str,label_relation))," ".join(map(str,label_node[0]))," ".join(map(str,properties_key[0]))," ".join(map(str,node_value[1]))))
          
          if len(label_relation)==0: 
             cypher=(""" MATCH p = shortestPath((n1:{0} {{{1}:'{2}'}})-[*]-(n2:{4} {{{5}:'{6}'}})) return p 
                      """.format(" ".join(map(str,label_node[0]))," ".join(map(str,properties_key[0]))," ".join(map(str,node_value[0]))," ".join(map(str,label_relation))," ".join(map(str,label_node[0]))," ".join(map(str,properties_key[0]))," ".join(map(str,node_value[1]))))
    return cypher
