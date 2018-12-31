import csv_builder as cb
import EF_computer as ec
import time

"""
embedding
"""

# total time = 2180s
# start_time = time.time()
# print('computing the embeddings... 2')
# cb.parallel_embeddings(name='_2.csv', distance = 2, mean = 2)
# time_embedding = (time.time() - start_time)
# print(" time_embeddings :", time_embedding)
#
# start_time = time.time()
# print('computing the embeddings... 6')
# cb.parallel_embeddings(name='_6.csv', distance = 6, mean = 2)
# time_embedding = (time.time() - start_time)
# print(" time_embeddings :", time_embedding)
#
# start_time = time.time()
# print('computing the embeddings... 8')
# cb.parallel_embeddings(name='_8.csv', distance = 8, mean = 2)
# time_embedding = (time.time() - start_time)
# print(" time_embeddings :", time_embedding)


"""
Enrichment task
"""

start_time = time.time()
print('computing the benchmark for 2')
ec.parallel_benchmark('2', percent=[0.25, 0.5, 1], name_embedding='_2.csv')
time_benchmark = (time.time() - start_time)
print(" time_benchmark :", time_benchmark)

start_time = time.time()
print('computing the benchmark for 6')
ec.parallel_benchmark('6', percent=[0.25, 0.5, 1], name_embedding='_6.csv')
time_benchmark = (time.time() - start_time)
print(" time_benchmark :", time_benchmark)

start_time = time.time()
print('computing the benchmark for 8')
ec.parallel_benchmark('8', percent=[0.25, 0.5, 1], name_embedding='_8.csv')
time_benchmark = (time.time() - start_time)
print(" time_benchmark :", time_benchmark)
