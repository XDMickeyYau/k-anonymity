from elasticsearch import Elasticsearch
from pandasticsearch import Select
import pandas as pd
from k_anonymity import *
import util
import os
script_dir = os.path.dirname(__file__)


'''get elastic search data as Python dict'''
from elasticsearch import Elasticsearch
es = Elasticsearch('http://localhost:9200')
result_dict = es.search(index="kibana_sample_data_logs", body={"query": {"match_all": {}}},size=10000)
df = Select.from_dict(result_dict).to_pandas()

'''data cleaning'''
df=util.explode(df,['tags'])
for column in df.select_dtypes('object').columns:
    df[column]=df[column].astype('category')
df['response']=df['response'].astype('category')
print("Data type of columns:")
print(df.dtypes)

'''analyse data span'''
full_spans = get_spans(df, df.index)
print("Full span:")
print(full_spans)

'''choose column to be anonymized'''
feature_columns = ['geo.coordinates.lat', 'geo.coordinates.lon']
sensitive_column = 'response'

'''grouping data so thateach group have k member'''
print("Start k-anonymity")
finished_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, is_k_anonymous)
print("Finished partition with length {}".format(len(finished_partitions)))

dfn = build_anonymized_dataset(df, finished_partitions, feature_columns, sensitive_column)
print(dfn.sort_values(feature_columns+[sensitive_column]))
Export = dfn.to_json (orient="table")
with open('output/k-anonymity.json', 'w+') as f_obj:
    f_obj.write(Export)

'''grouping data so thateach group have k member and l unique value'''
print("Start l-diversity")
finished_l_diverse_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, lambda *args: is_k_anonymous(*args) and is_l_diverse(*args))
print("Finished partition with length {}".format(len(finished_l_diverse_partitions)))

dfl = build_anonymized_dataset(df, finished_l_diverse_partitions, feature_columns, sensitive_column)
dfl.sort_values(feature_columns+[sensitive_column])
Export = dfl.to_json (orient="table")
with open('output/ l-diversity.json', 'w+') as f_obj:
    f_obj.write(Export)

'''grouping data so thateach group have k member, l unique value, and distribution close to the whole data'''
print("Start t-closeness")
# here we generate the global frequencies for the sensitive column
global_freqs = {}
total_count = float(len(df))
group_counts = df.groupby(sensitive_column)[sensitive_column].agg('count')
for value, count in group_counts.to_dict().items():
    p = count/total_count
    global_freqs[value] = p
print("Global frequency with length {}".format(global_freqs))

finished_t_close_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, lambda *args: is_k_anonymous(*args) and is_t_close(*args, global_freqs))
print("Finished partition with length {}".format(len(finished_t_close_partitions)))

dft = build_anonymized_dataset(df, finished_t_close_partitions, feature_columns, sensitive_column)
dft.sort_values(feature_columns+[sensitive_column])
Export = dft.to_json (orient="table")
with open('output/ t-closeness.json', 'w+') as f_obj:
    f_obj.write(Export)

