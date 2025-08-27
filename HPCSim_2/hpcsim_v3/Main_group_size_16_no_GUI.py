"""
Created on 2015/07/17

@author: smallcat

grid
"""

# from __future__ import division
# import tkinter
# tkinter._test()
import networkx as nx
import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
# import itertools
import threading
# from . import GUI
# import GUI
import PWA
import time
# import datetime
import numpy as np
import random
import re
import os
# from datetime import timedelta
from datetime import datetime, timedelta
import csv

# import Torus


# simulation constants
TIMESTEP_UNIT = 1
SPEED_UP_FACTOR =  10000.0# 100000.0
NUM_SIMULATION_JOBS = 50

# global variables
size_grid_x = 0
size_grid_y = 0
RG = None
all_jobs = None
job_queue = None
all_jobs_submitted = False
result_wait_sum = 0

arival_result_queue = None
start_result_queue = None

# fig, ax = plt.subplots(figsize=(6, 6))
fig, (ax_graph, ax_heatmap, ax_heatmap_64_64) = plt.subplots(1, 3, figsize=(18, 8), gridspec_kw={'width_ratios': [2, 1, 1]}) # 左を広く、右を狭く
fig_gantt, ax_gantt = plt.subplots(figsize=(14, 8))


#wait_times = []
#for i in range(1, num):
#    wait_times[i] = 0
"""
def update(frame):
    ax.clear()
    active_nodes = {}
    for job_id, (cores, runtime, _, _, start, _, nodes) in job_records:
        end = start + timedelta(seconds=runtime)
        if start <= frame < end:
            for n in nodes:
                active_nodes[n] = job_id

    node_colors = []
    labels = {}
    for n in G.nodes:
        if n in active_nodes:
            node_colors.append("red")
            labels[n] = str(active_nodes[n])
        else:
            node_colors.append("lightgray")
            labels[n] = ""

    nx.draw_networkx(RG, pos=pos, ax=ax, with_labels=True, labels=labels,
                     node_color=node_colors, node_size=500)
    ax.set_title(f"Time: {frame.strftime('%H:%M:%S')}")
    ax.axis("off")
"""

def generate_symmetric_matrix():
    while True:
        # x, y, z
        x = random.randint(0,16)
        y = random.randint(0,16 - x)
        z = 16 - x - y
        
        # 行2: x + w + u = 16
        a = 16 - x
        # 行3: y + w + v = 16
        b = 16 - y
        # 行4: z + u + v = 16
        c = 16 - z
        
        w = (a + b - c) / 2
        u = (a + c - b) / 2
        v = (b + c - a) / 2
        
        # 整数かつ非負か確認
        if any(val < 0 for val in [w,u,v]):
            continue
        if not all(float(val).is_integer() for val in [w,u,v]):
            continue
        
        w,u,v = int(w), int(u), int(v)
        
        # 完成
        A = np.array([
            [0, x, y, z],
            [x, 0, w, u],
            [y, w, 0, v],
            [z, u, v, 0]
        ], dtype=int)
        return A

def generate_symmetric_matrix_varying_rowsums(row_sums):
    while True:
        # x, y, z は行1の和で決める
        s1 = row_sums[0]
        x = random.randint(0, s1)
        y = random.randint(0, s1 - x)
        z = s1 - x - y
        
        # 残り
        a = row_sums[1] - x
        b = row_sums[2] - y
        c = row_sums[3] - z
        
        w = (a + b - c) / 2
        u = (a + c - b) / 2
        v = (b + c - a) / 2
        
        # 整数・非負
        if any(val < 0 for val in [w,u,v]):
            continue
        if not all(float(val).is_integer() for val in [w,u,v]):
            continue
        
        w,u,v = int(w), int(u), int(v)
        
        A = np.array([
            [0, x, y, z],
            [x, 0, w, u],
            [y, w, 0, v],
            [z, u, v, 0]
        ], dtype=int)
        
        return A
"""
def set_gui_parameters():
    app = GUI.MyApp(0)  # Create an instance of the application class
    app.MainLoop()  # Tell it to start processing events
"""

def initialize_graph():
    global size_grid_x, size_grid_y
    # set_gui_parameters()
    # torus_d = 1
    # xa = 5  # grid length
    # ya = 5  # grid width
    # size_grid_x = GUI.xxxx  # grid length
    # size_grid_y = GUI.yyyy  # grid width
    initialize_RG()


def initialize_RG():
    global RG, pos
    # RG = nx.Graph()
    # pos = dict(zip(RG,RG))

    # if(GUI.topo == "grid"):
    # RG = nx.grid_2d_graph(size_grid_x, size_grid_y)

    RG = nx.Graph()
    RG.add_nodes_from(range(0,69))
    # pos = dict(list(zip(RG, RG)))

    for i in range(64):
        RG.nodes[i]["ava"] = "yes"

    for i in range(0,64):
        RG.add_edge(i, (i // 16)  + 64)
    
    for i in range(64,68):
        RG.nodes[i]["group_large_ava"] = 8
        RG.add_edge(i,68)

    pos = nx.spring_layout(RG)

    #for i in range(size_grid_x):
    #    for j in range(size_grid_y):
    #        RG.nodes[(i, j)]["ava"] = "yes"  # node availabitily
    # else:
    #     torus_d = int(GUI.topo[0])
    #     if(torus_d == 2): #8*8
    #         RG = nx.grid_graph(dim=[8,8], periodic=True)
    # #         pos = dict(zip(RG,RG))
    #         for a in range(8):
    #             for b in range(8):
    # #                 RG.add_node((a,b))
    #                 RG.node[(a,b)]["ava"] = "yes"
    #     if(torus_d == 3): #8*8*8
    #         RG = nx.grid_graph(dim=[8,8,8], periodic=True)
    # #         pos = dict(zip(RG,RG))
    #         for a in range(8):
    #             for b in range(8):
    #                 for c in range(8):
    # #                     RG.add_node((a,b,c))
    #                     RG.node[(a,b,c)]["ava"] = "yes"
    #     if(torus_d == 4): #8*8*8*4
    #         RG = nx.grid_graph(dim=[8,8,8,4], periodic=True)
    # #         pos = dict(zip(RG,RG))
    #         for a in range(8):
    #             for b in range(8):
    #                 for c in range(8):
    #                     for d in range(4):
    # #                         RG.add_node((a,b,c,d))
    #                         RG.node[(a,b,c,d)]["ava"] = "yes"
    #     if(torus_d == 5): #8*8*8*4*4
    #         RG = nx.grid_graph(dim=[8,8,8,4,4], periodic=True)
    # #         pos = dict(zip(RG,RG))
    #         for a in range(8):
    #             for b in range(8):
    #                 for c in range(8):
    #                     for d in range(4):
    #                         for e in range(4):
    # #                             RG.add_node((a,b,c,d,e))
    #                             RG.node[(a,b,c,d,e)]["ava"] = "yes"


# nx.write_adjlist(RG,"test.adjlist")


def initialize_all_jobs_list():
    global all_jobs
    jobs = {}

    # jobs[0] = (4,8)   #job number, required cpus, required time (in seconds)
    # jobs[1] = (2,4)
    # jobs[2] = (6,8)
    # jobs[3] = (2,1)
    # jobs[4] = (4,4)
    # jobs[5] = (2,3)
    # jobs[6] = (4,8)
    # jobs[7] = (6,5)
    # jobs[8] = (1,7)
    # jobs[9] = (5,2)
    # jobs[10] = (5,6)
    # jobs[11] = (-1,2)
    data = PWA.data
    for i in range(len(data)):
        #     jobs[i] = (data["Requested Number of Processors"][i], data["Requested Time"][i]/k)
        # jobs[i] = (data["Number of Allocated Processors"][i], data["Run Time"][i] / SPEED_UP_FACTOR, data["Submit Time"][i] / SPEED_UP_FACTOR, "submited_in_sim_time", "started_in_sim_time", "traffic_matrix", "use_nodes", "64*64_traffic_matrix")
        jobs[i] = (int(data["Number of Allocated Processors"][i]), data["Run Time"][i] / SPEED_UP_FACTOR, data["Submit Time"][i] / SPEED_UP_FACTOR, -1, -1, -1, -1, -1)
    # jobs = []
    # jobs.append(0, (4,800))
    # jobs.append(1, (2,400))
    # jobs.append(2, (6,800))
    # jobs.append(3, (2,100))
    # jobs.append(4, (4,400))
    # jobs.append(5, (2,300))
    # jobs.append(6, (4,800))
    # jobs.append(7, (6,500))

    # jobs_ = zip(jobs.keys(), jobs.values())
    all_jobs = list(jobs.items())
    if all_jobs[0][1][0] != 1:
        np_val = all_jobs[0][1][0]
        pattern = re.compile(rf"flatten_matrix_(\w+)_A_{np_val}\.csv")
        benchmarks = []
        target_dir = "hpcsim_v3/traf_mat_flatten"
        for file in os.listdir(target_dir):
            match = pattern.match(file)
            if match:
                benchmarks.append(match.group(1))
        selected_benchmark = random.choice(benchmarks)
        selected_file = f"{target_dir}/flatten_matrix_{selected_benchmark}_A_{np_val}.csv"

        updated_job = (all_jobs[0][1][0], all_jobs[0][1][1], all_jobs[0][1][2], datetime.now(), -1, np.loadtxt(selected_file, delimiter=","), -1, np.loadtxt(selected_file, delimiter=","))
        all_jobs[0] = (0, updated_job)
    else:
        updated_job = (all_jobs[0][1][0], all_jobs[0][1][1], all_jobs[0][1][2], datetime.now(), -1, -1, -1, -1)
        all_jobs[0] = (0, updated_job)


def initialize_job_queue():
    global job_queue, all_jobs
    job_queue = [all_jobs[0]]
    # 150821 huyao available except FIFO
    """
    if (GUI.schedule == "LIFO"):
        job_queue.insert(0, job_queue.pop(-1))
    """


def submit_jobs():
    global all_jobs
    current_time = all_jobs[0][1][2]
    global job_queue, all_jobs_submitted
    """
    np_val = all_jobs[0][1][0]
    pattern = re.compile(rf"flatten_matrix_(\w+)_A_{np_val}\.csv")
    benchmarks = []
    target_dir = "hpcsim_v3/traf_mat_flatten"
    for file in os.listdir(target_dir):
        match = pattern.match(file)
        if match:
            benchmarks.append(match.group(1))
    selected_benchmark = random.choice(benchmarks)
    selected_file = f"{target_dir}/flatten_matrix_{selected_benchmark}_A_{np_val}.csv"

    updated_job = (all_jobs[0][1][0], all_jobs[0][1][1], all_jobs[0][1][2], datetime.datetime.now(), -1, np.loadtxt(selected_file, delimiter=","))
    all_jobs[0] = (0, updated_job)
    """
    for i in range(1, NUM_SIMULATION_JOBS):
        wait_time = all_jobs[i][1][2] - current_time
        if (wait_time >= 0):
            # time.sleep(wait_time)
            current_time = all_jobs[i][1][2]

            if all_jobs[i][1][0] != 1:
                np_val = all_jobs[i][1][0]
                pattern = re.compile(rf"flatten_matrix_(\w+)_A_{np_val}\.csv")
                benchmarks = []
                target_dir = "hpcsim_v3/traf_mat_flatten"
                for file in os.listdir(target_dir):
                    match = pattern.match(file)
                    if match:
                        benchmarks.append(match.group(1))
                selected_benchmark = random.choice(benchmarks)
                selected_file = f"{target_dir}/flatten_matrix_{selected_benchmark}_A_{np_val}.csv"

                updated_job = (all_jobs[i][1][0], all_jobs[i][1][1], all_jobs[i][1][2], datetime.now(), -1, np.loadtxt(selected_file, delimiter=","), -1, np.loadtxt(selected_file, delimiter=","))
                all_jobs[i] = (i, updated_job)
                job_queue.append(all_jobs[i])
            else:
                updated_job = (all_jobs[i][1][0], all_jobs[i][1][1], all_jobs[i][1][2], datetime.now(), -1, -1, -1, -1)
                all_jobs[i] = (i, updated_job)
                job_queue.append(all_jobs[i])
            #arival_result_queue.append(datetime.datetime.now)
            

            # 150824 huyao
            #             global lock
            #             lock = True
            """
            if (GUI.schedule == "BF"):
                # 150824 huyao first unchanged during insertion
                one = job_queue.pop(0)
                job_queue = sorted(job_queue, key=lambda abc: abc[1], reverse=True)
                job_queue.insert(0, one)
            if (GUI.schedule == "SF"):
                one = job_queue.pop(0)
                job_queue = sorted(job_queue, key=lambda abc: abc[1])
                job_queue.insert(0, one)
            #             if(GUI.schedule == "LIFO"):
            #                 queue.insert(1, queue.pop(-1))
            #             lock = False
            """

            print(datetime.now(), "job: ", all_jobs[i], " is submitted")
        else:
            print(datetime.now(), "job: ", all_jobs[i], " can not be submitted")
            i = i + 1
        if (i == NUM_SIMULATION_JOBS - 1):
            all_jobs_submitted = True


# jobs_ = jobs.items()    #dic -> tuple
# if(GUI.schedule == "BF"):
#     jobs_ = sorted(jobs.items(), key=lambda abc:abc[1], reverse=True)
# if(GUI.schedule == "SF"):
#     jobs_ = sorted(jobs.items(), key=lambda abc:abc[1])   

# print jobs_ 

# print RG.nodes(data = True)
# print RG.edges()


# print jobs_[0][1][0]

# get all sub-graphs with x nodes
# target = nx.complete_graph(jobs_[0][1][0])
# for sub_nodes in itertools.combinations(RG.nodes(),len(target.nodes())):
#     subg = RG.subgraph(sub_nodes)
#     if nx.is_connected(subg):
#         #print subg.edges()
#         print subg.nodes()

# jobs_.pop(0)
# print jobs_

# jobs.pop(jobs_[0][0])
# print jobs

# def printt(str):
#     print datetime.datetime.now(), str


def divi(n, start=2):
    if (n == 1):
        return 1, 1
    for i in range(start, n + 1):
        if (n % i == 0):
            return i, int(n / i)  # i width; n/i length


# def unlock(node, endx, endy, endxx, endyy, job): 
#     node["ava"] = "yes"
#     nodelist.remove((endx,endy))
#     if(endx==endxx and endy==endyy):
#         print datetime.datetime.now(), "job: ", job, "is finished"

def unlock_unavailable(nl, job, traffic_matrix, traffic_matrix_64_64):
    global current_traffic_matrix, current_traffic_matrix_64_64
    while (len(nl) > 0):
        RG.nodes[nl[0]]["ava"] = "yes"
        nodelist_to_draw.remove(nl[0])
        nl.pop(0)
    if isinstance(traffic_matrix, np.ndarray):
        current_traffic_matrix = current_traffic_matrix - traffic_matrix
        current_traffic_matrix_64_64 = current_traffic_matrix_64_64 -traffic_matrix_64_64
    #print(datetime.datetime.now(), "job: ", job, "is finished")


#     checkover()

# def unlock0(node, x, y, i, job): 
#     node["ava"] = "yes"
#     nodelist.remove((x,y))
#     if(i==0):
#         print datetime.datetime.now(), "job: ", job, "is finished"
# 150825 huyao

def qubo_allocation(first_job, first_job_cpu, first_job_time):
    global RG, all_jobs
    # you can do anything
    # return True if you have successfully allocated the first job
    return False


def fso(first_job, first_job_cpu, first_job_time):
    count = 0
    ava_nodes = []
    # global fso_not_found
    for yy in range(size_grid_y):
        for xx in range(size_grid_x):
            if (RG.nodes[(xx, yy)]["ava"] == "yes"):
                count = count + 1
                ava_nodes.append((xx, yy))
                if (count == first_job_cpu):
                    print(datetime.now(), "job: ", first_job, " is scheduled to the nodes (fso):")
                    for i in range(len(ava_nodes)):
                        RG.nodes[ava_nodes[i]]["ava"] = "no"
                        #                         print "(", ava_nodes[i][0], ", ", ava_nodes[i][1], ") "
                        nodelist_to_draw.append(ava_nodes[i])
                    # t = threading.Timer(jobs_[0][1][1], unlock0, (RG.node[ava_nodes[i]], ava_nodes[i][0],
                    # ava_nodes[i][1], i, jobs_[0],)) #required processing time
                    # t.start()
                    t = threading.Timer(first_job_time, unlock_unavailable, (ava_nodes, first_job,))  # required processing time
                    t.start()
                    job_queue.pop(0)
                    #                     fill = 0
                    # fso_not_found = False
                    # return
                    return False
            # if (xx == size_grid_x - 1 and yy == size_grid_y - 1):
            #     fso_not_found = True
    return True


#     lock = False

# stopwrite = False

# def checkover():
#     if(len(queue)<1):
#         global stopwrite
#         stopwrite = True 



def dostat():
    # global timestep
    # timestep = 0.5     150826 huyao scheduling->pwa  0.5->1
    # timestep = TIMESTEP
    global all_jobs

    wall_time_step = 0  # time step
    total_nodes_num = size_grid_x * size_grid_y  # total nodes
    #     f = open("stat_su", "w") #system utilization
    date_time_now = str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-")
    # date_time_now = date_time_now.replace(" ", "-")
    # date_time_now = date_time_now.replace(".", "-")
    # date_time_now = date_time_now.replace(":", "-")
    archive_name = PWA.archive.replace(".", "-")
    file_name = "stat_su_" + date_time_now + "_" + str(size_grid_x * size_grid_y) + "_" + "schedule" + "_" + "mode" + "_" + archive_name  # file name
    # file_name = "stat_su_" + date_time_now + "_" + str(size_grid_x * size_grid_y) + "_" + GUI.schedule + "_" + GUI.mode + "_" + archive_name  # file name
    file_handle = open(file_name, "w")  # system utilization
    file_handle.write("#timestep  occupied  total  utilization\n")
    file_handle.close()
    #     while(stopwrite==False):
    while (True):
        total_time_step = 0
        for ax in range(size_grid_y):
            for ay in range(size_grid_x):
                if (RG.nodes[(ax, ay)]["ava"] == "no"):
                    total_time_step = total_time_step + TIMESTEP_UNIT
        wall_time_step = wall_time_step + TIMESTEP_UNIT
        #         print ts, "    ", total/tn
        log_message = (str(wall_time_step * TIMESTEP_UNIT) + "    " + str(total_time_step) + "    " + str(total_nodes_num) + "    "
                       + str(float(total_time_step) / total_nodes_num) + "\n")  # 150826 huyao ts->ts*timestep
        file_handle = open(file_name, "a")  # system utilization
        file_handle.write(log_message)
        file_handle.close()
        time.sleep(TIMESTEP_UNIT)
        if (total_time_step == 0 and len(job_queue) == 0 and all_jobs_submitted == True):
            break


def start_jobs_submitting_thread():
    sj = threading.Timer(0, submit_jobs)  # submit jobs
    sj.start()


def start_dostat_thread():
    stat = threading.Timer(0, dostat)  # required processing time
    stat.start()

nodelist_to_draw = []

def loop_allocate_all_jobs():
    global result_wait_sum
    global result_utility
    global all_jobs
    global current_traffic_matrix, current_traffic_matrix_64_64
 
    fso_not_found = False
    # global to_first
    # 150825 huyao to_first in normal = fso_not_found in fso
    to_first = True

    # global first, first_num, first_cpu, first_time
    first = job_queue[0]
    first_num = job_queue[0][0]
    first_cpu = job_queue[0][1][0]
    first_time = job_queue[0][1][1]

    #
    x = 0
    y = 0
    transform = False
    temp_x = 0
    temp_y = 0
    fill = 0
    node_count = 0
    result_utility = 0

    G0_capa = 16.0
    G1_capa = 16.0
    G2_capa = 16.0
    G3_capa = 16.0

    G0_lock = 0.0
    G1_lock = 0.0
    G2_lock = 0.0
    G3_lock = 0.0

    capa_list = [int(G0_capa), int(G1_capa), int(G2_capa), int(G3_capa)]
    lock_list = [G0_lock, G1_lock, G2_lock, G3_lock]

    connect_matrix = generate_symmetric_matrix_varying_rowsums(capa_list)

    current_traffic_matrix = np.zeros((4, 4), dtype=float)
    current_traffic_matrix_64_64 = np.zeros((64, 64), dtype=float)

    group_idle_counts = []
    group_large_ava_counts = []

    node = [0] * 64

    for i in range(64):
        node[i] = 0

    # 150821 huyao jobs_->queue  avoid any other job inserted during transform
    # lock = False

    def reset():
        # global x, y, fill, transform, tempX, tempY
        nonlocal x, y, fill, transform , temp_x, temp_y
        x = 0
        y = 0
        transform = False
        temp_x = 0
        temp_y = 0
        fill = 0


    # while(len(queue)>0):
    while (True):
        """
        
        本来ここにutilizationじゃね？
        
        """

        if (len(job_queue) > 0):
            """
            ワンチャンここでサブミットしたい
            """



            """
            工事中
            """

            result_wait_sum = result_wait_sum + 1

            for i in range(64):
                if RG.nodes[i]["ava"] == "no":
                    result_utility = result_utility + 1

            #     print "haha", jobs_[0]
            #     print x
            #     print y
            #     print transform
            #     print tempX
            #     print tempY
            #     print fill

            # 150821 huyao available except FIFO
            #         if(lock==False):

            node_count = 0
            for i in range(64):
                if RG.nodes[i]["ava"] == "yes":
                    node_count += 1
            first = job_queue[0]
            first_num = job_queue[0][0]
            first_cpu = job_queue[0][1][0]
            first_time = job_queue[0][1][1]
            first_traffic_matrix = job_queue[0][1][5]

            if (first_cpu > 64):
                first_cpu = 64

            ava_to_unava = []
            
            # 最弱制約
            """
            if (node_count >= first_cpu):
                for i in range(first_cpu):
                    for j in range(64):
                        if RG.nodes[j]["ava"] == "yes":
                            RG.nodes[j]["ava"] = "no"
                            nodelist_to_draw.append(j)
                            ava_to_unava.append(j)
                            break
                        else:
                            continue
                t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava, first,))  # required processing time
                t.start()

                print(job_queue[0])
                 
                updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.datetime.now(), -1)
                all_jobs[first_num] = (first_num, updated_job_done)
                print(job_queue[0])
                job_queue.pop(0)
            else:
                continue
            """
            # 最弱制約

            # schedule制約最大強化

            """

            繋いでいるところに置く
            > 空いている中でランダムに繋ぐ (networkx)
            > ランダムに繋いだところでスケジュールを頑張る(GRAP)

            """
            """

            if (node_count >= first_cpu):

                for i in range(first_cpu):
                    for j in range(64):
                        if RG.nodes[j]["ava"] == "yes":
                            RG.nodes[j]["ava"] = "no"
                            nodelist_to_draw.append(j)
                            ava_to_unava.append(j)
                            break
                        else:
                            continue
                t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava, first,))  # required processing time
                t.start()

                print(job_queue[0])
                 
                updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.datetime.now(), -1)
                all_jobs[first_num] = (first_num, updated_job_done)
                print(job_queue[0])
                job_queue.pop(0)
            else:
                continue

            """
            # schedule制約最大強化

            # reduced round robin



            # reduced round robin

            # round robin
            """

            if (node_count >= first_cpu):
                for i in range(first_cpu):
                    for j in range(64):
                        if RG.nodes[j]["ava"] == "yes":
                            RG.nodes[j]["ava"] = "no"
                            nodelist_to_draw.append(j)
                            ava_to_unava.append(j)
                            break
                        else:
                            continue
                t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava, first,))  # required processing time
                t.start()

                print(job_queue[0])
                 
                updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.datetime.now(), -1)
                all_jobs[first_num] = (first_num, updated_job_done)
                print(job_queue[0])
                job_queue.pop(0)
            else:
                continue

            """
            # round robin

            # 自由OCS

            # 自由OCS終了

            # 絶対Local Fat

            """

            group_idle_counts = []
            # group_large_ava_counts = []
            if (node_count >= first_cpu and first_cpu <= 16):
                for group in range(4):
                    idle_count = 0
                    for node_num in range(group * 16,(group + 1) * 16):
                        if RG.nodes[node_num]["ava"] == "yes":
                            idle_count += 1
                    group_idle_counts.append(idle_count)
                filtered_counts = [count_ for count_ in group_idle_counts if count_ >= first_cpu ]

                if filtered_counts:
                    min_idle = min(filtered_counts)
                    candidate_groups = [i_ for i_ , cnt in enumerate(group_idle_counts) if cnt == min_idle and cnt >= first_cpu]
                    selected_group = candidate_groups[0]
                    idle_nodes = [i for i in range(selected_group*16, (selected_group+1)*16) if RG.nodes[i]["ava"] == "yes"]
                    chosen_idle_nodes = [idle_nodes[i] for i in range(first_cpu)]
                    print(first)
                    for chosen_node in chosen_idle_nodes:
                        RG.nodes[chosen_node]["ava"] = "no"
                        nodelist_to_draw.append(chosen_node)
                        ava_to_unava.append(chosen_node)
                    ava_to_unava_new_list = ava_to_unava

                    if first_cpu != 1:
                        traffic_matrix = job_queue[0][1][5]
                        order = np.argsort(ava_to_unava)
                        old_to_new = np.empty_like(order)
                        old_to_new[order] = np.arange(len(ava_to_unava))
                        new_pos = [0] * len(old_to_new)
                        for old_idx, new_idx in enumerate(old_to_new):
                            new_pos[new_idx] = old_idx
                        traffic_matrix_permutated = traffic_matrix[new_pos, :][:, new_pos]
                        new_list = [None] * len(traffic_matrix)
                        for old_idx, new_idx in enumerate(old_to_new):
                            new_list[new_idx] = ava_to_unava[old_idx]
                        print(f"ava_to_unava: {ava_to_unava}")
                        print(f"new_list: {new_list}")
                        G0_ava_to_unava_list = []
                        G1_ava_to_unava_list = []
                        G2_ava_to_unava_list = []
                        G3_ava_to_unava_list = []
                        group_ava_to_unava = [G0_ava_to_unava_list, G1_ava_to_unava_list, G2_ava_to_unava_list, G3_ava_to_unava_list]
                        for i in range(len(new_list)):
                            if new_list[i] <= 15:
                                G0_ava_to_unava_list.append(i)
                            elif(new_list[i] >= 16) and (new_list[i] <= 31):
                                G1_ava_to_unava_list.append(i)
                            elif(new_list[i] >= 32) and (new_list[i] <= 47):
                                G2_ava_to_unava_list.append(i)
                            else:
                                G3_ava_to_unava_list.append(i)
                        matrix_reduced = 0
                        print(group_ava_to_unava)
                        if len(G0_ava_to_unava_list) >= 2:
                            traffic_matrix_permutated[[G0_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G0_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G0_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G0_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G0_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G0_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            matrix_reduced += (len(G0_ava_to_unava_list) - 1)
                        if len(G1_ava_to_unava_list) >= 2:
                            G1_ava_to_unava_list = [x - matrix_reduced for x in G1_ava_to_unava_list]
                            traffic_matrix_permutated[[G1_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G1_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G1_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G1_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G1_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G1_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            matrix_reduced += (len(G1_ava_to_unava_list) - 1)
                        if len(G2_ava_to_unava_list) >= 2:
                            G2_ava_to_unava_list = [x - matrix_reduced for x in G2_ava_to_unava_list]
                            traffic_matrix_permutated[[G2_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G2_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G2_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G2_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G2_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G2_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            matrix_reduced += (len(G2_ava_to_unava_list) - 1)
                        if len(G3_ava_to_unava_list) >= 2:
                            G3_ava_to_unava_list = [x - matrix_reduced for x in G3_ava_to_unava_list]
                            traffic_matrix_permutated[[G3_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G3_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G3_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G3_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G3_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G3_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                        for i in range(len(group_ava_to_unava)):
                            if len(group_ava_to_unava[i]) == 0:
                                zero_row = np.zeros((1, traffic_matrix_permutated.shape[1]), dtype=traffic_matrix_permutated.dtype)
                                traffic_matrix_permutated_row_sorted = np.vstack((traffic_matrix_permutated[:i, :], zero_row, traffic_matrix_permutated[i:, :]))
                                zero_col = np.zeros((traffic_matrix_permutated_row_sorted.shape[0], 1), dtype=traffic_matrix_permutated_row_sorted.dtype)
                                traffic_matrix_permutated_row_col_sorted = np.hstack((traffic_matrix_permutated_row_sorted[:, :i], zero_col, traffic_matrix_permutated_row_sorted[:, i:]))
                                traffic_matrix_permutated = traffic_matrix_permutated_row_col_sorted
                        
                        current_traffic_matrix = current_traffic_matrix + traffic_matrix_permutated
                        #print(job_queue[0])
                        
                        updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), traffic_matrix_permutated, ava_to_unava)
                        all_jobs[first_num] = (first_num, updated_job_done)

                        t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                        t.start()

                        #print(job_queue[0])
                        job_queue.pop(0)
                        print(f"group_idle_counts: {group_idle_counts}")
                    else:
                        updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), -1, ava_to_unava)
                        #print("HERE!!")
                        all_jobs[first_num] = (first_num, updated_job_done)

                        t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                        t.start()

                        #print(job_queue[0])
                        job_queue.pop(0)
                        print(f"group_idle_counts: {group_idle_counts}")

                    # t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava, first,))  # required processing time
                    # t.start()
                    # updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.datetime.now(), -1)
                    # all_jobs[first_num] = (first_num, updated_job_done)
                    # job_queue.pop(0)
                    continue
                else:
                    continue
            elif (node_count >= first_cpu and first_cpu > 16):
                #print(first_cpu)
                group_idle_counts = []
                for group in range(4):
                    idle_count = 0
                    for node_num in range(group * 16,(group + 1) * 16):
                        if RG.nodes[node_num]["ava"] == "yes":
                            idle_count += 1
                    group_idle_counts.append(idle_count)
                print(f"group_idle_counts: {group_idle_counts}")
                sorted_group_idle_counts = sorted(group_idle_counts, reverse=True)
                print(f"sorted_group_idle_counts: {sorted_group_idle_counts}")
                print(f"group_idle_counts: {group_idle_counts}")
                use_group_num = first_cpu // 16
                if first_cpu % 16 != 0:
                    use_group_num += 1
                for i in range(use_group_num):
                    if ((i + 1) * sorted_group_idle_counts[i] >= first_cpu):
                        print(first)
                        print(i)
                        one_rack_cpu = first_cpu // (i + 1)
                        one_rack_cpu_arr = [one_rack_cpu for n in range(i + 1)]
                        one_rack_cpu_ex = first_cpu % (i + 1)
                        for j in range(one_rack_cpu_ex):
                            one_rack_cpu_arr[j] = one_rack_cpu_arr[j] + 1
                        print(first_num)
                        print(f"sorted_group_idle_counts: {sorted_group_idle_counts}")
                        print(f"group_idle_counts: {group_idle_counts}")
                        print(f"one_rack_cpu_arr: {one_rack_cpu_arr}")
                        print(f"group_idle_counts: {group_idle_counts}")
                        for n in one_rack_cpu_arr:
                            filtered_counts = [count_ for count_ in group_idle_counts if count_ >= n ]
                            if filtered_counts:
                                min_idle = min(filtered_counts)
                                candidate_groups = [i_ for i_ , cnt in enumerate(group_idle_counts) if cnt == min_idle and cnt >= n]
                                selected_group = candidate_groups[0]
                                idle_nodes = [i_ for i_ in range(selected_group*16, (selected_group+1)*16) if RG.nodes[i_]["ava"] == "yes"]
                                chosen_idle_nodes = [idle_nodes[i] for i in range(int(n))]
                                print(chosen_idle_nodes)
                                for chosen_node in chosen_idle_nodes:
                                    RG.nodes[chosen_node]["ava"] = "no"
                                    nodelist_to_draw.append(chosen_node)
                                    ava_to_unava.append(chosen_node)
                                group_idle_counts[selected_group] = group_idle_counts[selected_group] - n
                                print(group_idle_counts)
                            else:
                                print("エラー")
                                exit()
                                continue
                        # t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava, first,))  # required processing time
                        # t.start()
                        # print(job_queue[0]) 
                        # updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.datetime.now(), -1)
                        # all_jobs[first_num] = (first_num, updated_job_done)
                        # job_queue.pop(0)
                        if first_cpu != 1:
                            traffic_matrix = job_queue[0][1][5]
                            order = np.argsort(ava_to_unava)
                            old_to_new = np.empty_like(order)
                            old_to_new[order] = np.arange(len(ava_to_unava))
                            new_pos = [0] * len(old_to_new)
                            for old_idx, new_idx in enumerate(old_to_new):
                                new_pos[new_idx] = old_idx
                            traffic_matrix_permutated = traffic_matrix[new_pos, :][:, new_pos]
                            print(f"len(ava_to_unava): {len(ava_to_unava)}")
                            print(f"first_cpu: {first_cpu}")
                            print(f"len(traffic_matrix): {len(traffic_matrix)}")
                            new_list = [None] * len(traffic_matrix)
                            for old_idx, new_idx in enumerate(old_to_new):
                                new_list[new_idx] = ava_to_unava[old_idx]
                            print(new_list)
                            G0_ava_to_unava_list = []
                            G1_ava_to_unava_list = []
                            G2_ava_to_unava_list = []
                            G3_ava_to_unava_list = []
                            group_ava_to_unava = [G0_ava_to_unava_list, G1_ava_to_unava_list, G2_ava_to_unava_list, G3_ava_to_unava_list]
                            for i in range(len(new_list)):
                                if new_list[i] <= 15:
                                    G0_ava_to_unava_list.append(i)
                                elif(new_list[i] >= 16) and (new_list[i] <= 31):
                                    G1_ava_to_unava_list.append(i)
                                elif(new_list[i] >= 32) and (new_list[i] <= 47):
                                    G2_ava_to_unava_list.append(i)
                                else:
                                    G3_ava_to_unava_list.append(i)
                            matrix_reduced = 0
                            print(group_ava_to_unava)
                            if len(G0_ava_to_unava_list) >= 2:
                                traffic_matrix_permutated[[G0_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G0_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G0_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G0_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G0_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G0_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G0_ava_to_unava_list) - 1)
                            if len(G1_ava_to_unava_list) >= 2:
                                G1_ava_to_unava_list = [x - matrix_reduced for x in G1_ava_to_unava_list]
                                traffic_matrix_permutated[[G1_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G1_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G1_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G1_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G1_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G1_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G1_ava_to_unava_list) - 1)
                            if len(G2_ava_to_unava_list) >= 2:
                                G2_ava_to_unava_list = [x - matrix_reduced for x in G2_ava_to_unava_list]
                                traffic_matrix_permutated[[G2_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G2_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G2_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G2_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G2_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G2_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G2_ava_to_unava_list) - 1)
                            if len(G3_ava_to_unava_list) >= 2:
                                G3_ava_to_unava_list = [x - matrix_reduced for x in G3_ava_to_unava_list]
                                traffic_matrix_permutated[[G3_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G3_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G3_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G3_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G3_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G3_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            for i in range(len(group_ava_to_unava)):
                                if len(group_ava_to_unava[i]) == 0:
                                    zero_row = np.zeros((1, traffic_matrix_permutated.shape[1]), dtype=traffic_matrix_permutated.dtype)
                                    traffic_matrix_permutated_row_sorted = np.vstack((traffic_matrix_permutated[:i, :], zero_row, traffic_matrix_permutated[i:, :]))
                                    zero_col = np.zeros((traffic_matrix_permutated_row_sorted.shape[0], 1), dtype=traffic_matrix_permutated_row_sorted.dtype)
                                    traffic_matrix_permutated_row_col_sorted = np.hstack((traffic_matrix_permutated_row_sorted[:, :i], zero_col, traffic_matrix_permutated_row_sorted[:, i:]))
                                    traffic_matrix_permutated = traffic_matrix_permutated_row_col_sorted
                            
                            current_traffic_matrix = current_traffic_matrix + traffic_matrix_permutated
                            #print(job_queue[0])
                            
                            updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), traffic_matrix_permutated, ava_to_unava)
                            all_jobs[first_num] = (first_num, updated_job_done)

                            t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                            t.start()

                            #print(job_queue[0])
                            job_queue.pop(0)
                            # print(f"group_idle_counts: {group_idle_counts}")
                        else:
                            updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), -1, ava_to_unava)
                            #print("HERE!!")
                            all_jobs[first_num] = (first_num, updated_job_done)

                            t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                            t.start()

                            #print(job_queue[0])
                            job_queue.pop(0)
                            # print(f"group_idle_counts: {group_idle_counts}")
                        break
                    else:
                        continue
            else:
                continue

            """    

            # 絶対Local Fat 終了

            # 消極緩和Local Fat

            """

            group_idle_counts = []
            # group_large_ava_counts = []
            if (node_count >= first_cpu and first_cpu <= 16):
                for group in range(4):
                    idle_count = 0
                    for node_num in range(group * 16,(group + 1) * 16):
                        if RG.nodes[node_num]["ava"] == "yes":
                            idle_count += 1
                    group_idle_counts.append(idle_count)
                filtered_counts = [count_ for count_ in group_idle_counts if count_ >= first_cpu ]
                print(group_idle_counts)
                if filtered_counts:
                    min_idle = min(filtered_counts)
                    candidate_groups = [i_ for i_ , cnt in enumerate(group_idle_counts) if cnt == min_idle and cnt >= first_cpu]
                    selected_group = candidate_groups[0]
                    idle_nodes = [i for i in range(selected_group*16, (selected_group+1)*16) if RG.nodes[i]["ava"] == "yes"]
                    chosen_idle_nodes = [idle_nodes[i] for i in range(first_cpu)]
                    for chosen_node in chosen_idle_nodes:
                        RG.nodes[chosen_node]["ava"] = "no"
                        nodelist_to_draw.append(chosen_node)
                        ava_to_unava.append(chosen_node)
                    ava_to_unava_new_list = ava_to_unava

                    if first_cpu != 1:
                        traffic_matrix = job_queue[0][1][5]
                        order = np.argsort(ava_to_unava)
                        old_to_new = np.empty_like(order)
                        old_to_new[order] = np.arange(len(ava_to_unava))
                        new_pos = [0] * len(old_to_new)
                        for old_idx, new_idx in enumerate(old_to_new):
                            new_pos[new_idx] = old_idx
                        traffic_matrix_permutated = traffic_matrix[new_pos, :][:, new_pos]
                        new_list = [None] * len(traffic_matrix)
                        for old_idx, new_idx in enumerate(old_to_new):
                            new_list[new_idx] = ava_to_unava[old_idx]
                        #print(f"ava_to_unava: {ava_to_unava}")
                        #print(f"new_list: {new_list}")
                        G0_ava_to_unava_list = []
                        G1_ava_to_unava_list = []
                        G2_ava_to_unava_list = []
                        G3_ava_to_unava_list = []
                        group_ava_to_unava = [G0_ava_to_unava_list, G1_ava_to_unava_list, G2_ava_to_unava_list, G3_ava_to_unava_list]
                        for i in range(len(new_list)):
                            if new_list[i] <= 15:
                                G0_ava_to_unava_list.append(i)
                            elif(new_list[i] >= 16) and (new_list[i] <= 31):
                                G1_ava_to_unava_list.append(i)
                            elif(new_list[i] >= 32) and (new_list[i] <= 47):
                                G2_ava_to_unava_list.append(i)
                            else:
                                G3_ava_to_unava_list.append(i)
                        matrix_reduced = 0
                        #print(group_ava_to_unava)
                        if len(G0_ava_to_unava_list) >= 2:
                            traffic_matrix_permutated[[G0_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G0_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G0_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G0_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G0_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G0_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            matrix_reduced += (len(G0_ava_to_unava_list) - 1)
                        if len(G1_ava_to_unava_list) >= 2:
                            G1_ava_to_unava_list = [x - matrix_reduced for x in G1_ava_to_unava_list]
                            traffic_matrix_permutated[[G1_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G1_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G1_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G1_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G1_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G1_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            matrix_reduced += (len(G1_ava_to_unava_list) - 1)
                        if len(G2_ava_to_unava_list) >= 2:
                            G2_ava_to_unava_list = [x - matrix_reduced for x in G2_ava_to_unava_list]
                            traffic_matrix_permutated[[G2_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G2_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G2_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G2_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G2_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G2_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            matrix_reduced += (len(G2_ava_to_unava_list) - 1)
                        if len(G3_ava_to_unava_list) >= 2:
                            G3_ava_to_unava_list = [x - matrix_reduced for x in G3_ava_to_unava_list]
                            traffic_matrix_permutated[[G3_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G3_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G3_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G3_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G3_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G3_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                        for i in range(len(group_ava_to_unava)):
                            if len(group_ava_to_unava[i]) == 0:
                                zero_row = np.zeros((1, traffic_matrix_permutated.shape[1]), dtype=traffic_matrix_permutated.dtype)
                                traffic_matrix_permutated_row_sorted = np.vstack((traffic_matrix_permutated[:i, :], zero_row, traffic_matrix_permutated[i:, :]))
                                zero_col = np.zeros((traffic_matrix_permutated_row_sorted.shape[0], 1), dtype=traffic_matrix_permutated_row_sorted.dtype)
                                traffic_matrix_permutated_row_col_sorted = np.hstack((traffic_matrix_permutated_row_sorted[:, :i], zero_col, traffic_matrix_permutated_row_sorted[:, i:]))
                                traffic_matrix_permutated = traffic_matrix_permutated_row_col_sorted
                        
                        current_traffic_matrix = current_traffic_matrix + traffic_matrix_permutated
                        #print(job_queue[0])
                        
                        updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), traffic_matrix_permutated, ava_to_unava)
                        all_jobs[first_num] = (first_num, updated_job_done)

                        t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                        t.start()

                        #print(job_queue[0])
                        job_queue.pop(0)
                        print(f"group_idle_counts: {group_idle_counts}")
                    else:
                        updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), -1, ava_to_unava)
                        #print("HERE!!")
                        all_jobs[first_num] = (first_num, updated_job_done)

                        t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                        t.start()

                        #print(job_queue[0])
                        job_queue.pop(0)
                        print(f"group_idle_counts: {group_idle_counts}")

                    # t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava, first,))  # required processing time
                    # t.start()
                    # updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.datetime.now(), -1)
                    # all_jobs[first_num] = (first_num, updated_job_done)
                    # job_queue.pop(0)
                    continue
                else:
                    sorted_group_idle_counts = sorted(group_idle_counts, reverse = True)
                    print("sorted_group_idle_counts")
                    print(sorted_group_idle_counts)
                    for i in range(4):
                        if ((i + 1) * sorted_group_idle_counts[i] >= first_cpu):
                            one_rack_cpu = first_cpu // (i + 1)
                            one_rack_cpu_arr = [one_rack_cpu for n in range(i + 1)]
                            one_rack_cpu_ex = first_cpu % (i + 1)
                            for j in range(one_rack_cpu_ex):
                                one_rack_cpu_arr[j] = one_rack_cpu_arr[j] + 1
                            one_rack_cpu_arr_reversed = sorted(one_rack_cpu_arr, reverse = True)
                            for n in one_rack_cpu_arr_reversed:
                                filtered_counts = [count_ for count_ in group_idle_counts if count_ >= n]
                                if filtered_counts:
                                    min_idle = min(filtered_counts)
                                    candidate_groups = [i_ for i_ , cnt in enumerate(group_idle_counts) if cnt == min_idle and cnt >= n]
                                    selected_group = candidate_groups[0]
                                    idle_nodes = [i_ for i_ in range(selected_group*16, (selected_group+1)*16) if RG.nodes[i_]["ava"] == "yes"]
                                    chosen_idle_nodes = [idle_nodes[i] for i in range(int(n))]
                                    for chosen_node in chosen_idle_nodes:
                                        RG.nodes[chosen_node]["ava"] = "no"
                                        nodelist_to_draw.append(chosen_node)
                                        ava_to_unava.append(chosen_node)
                                    print(ava_to_unava)
                                    group_idle_counts[selected_group] = group_idle_counts[selected_group] - n
                                else:
                                    continue
                            if first_cpu != 1:
                                #print(first_cpu)
                                traffic_matrix = job_queue[0][1][5]
                                # print(traffic_matrix)
                                print(f"job_info: {first_cpu}")
                                print(f"len(traffic_matrix): {len(traffic_matrix)}")
                                print(f"ava_to_unava: {ava_to_unava}")
                                order = np.argsort(ava_to_unava)
                                old_to_new = np.empty_like(order)
                                old_to_new[order] = np.arange(len(ava_to_unava))
                                new_pos = [0] * len(old_to_new)
                                for old_idx, new_idx in enumerate(old_to_new):
                                    new_pos[new_idx] = old_idx
                                traffic_matrix_permutated = traffic_matrix[new_pos, :][:, new_pos]
                                new_list = [None] * len(traffic_matrix)
                                for old_idx, new_idx in enumerate(old_to_new):
                                    new_list[new_idx] = ava_to_unava[old_idx]
                                print(new_list)
                                G0_ava_to_unava_list = []
                                G1_ava_to_unava_list = []
                                G2_ava_to_unava_list = []
                                G3_ava_to_unava_list = []
                                group_ava_to_unava = [G0_ava_to_unava_list, G1_ava_to_unava_list, G2_ava_to_unava_list, G3_ava_to_unava_list]
                                for i in range(len(new_list)):
                                    if new_list[i] <= 15:
                                        G0_ava_to_unava_list.append(i)
                                    elif(new_list[i] >= 16) and (new_list[i] <= 31):
                                        G1_ava_to_unava_list.append(i)
                                    elif(new_list[i] >= 32) and (new_list[i] <= 47):
                                        G2_ava_to_unava_list.append(i)
                                    else:
                                        G3_ava_to_unava_list.append(i)
                                matrix_reduced = 0
                                print(group_ava_to_unava)
                                if len(G0_ava_to_unava_list) >= 2:
                                    traffic_matrix_permutated[[G0_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G0_ava_to_unava_list, :].sum(axis = 0)
                                    rows_to_delete = G0_ava_to_unava_list[1:]
                                    traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                    traffic_matrix_permutated_row_reduced[:,G0_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G0_ava_to_unava_list].sum(axis=1)
                                    cols_to_delete = G0_ava_to_unava_list[1:]
                                    traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                    traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                    matrix_reduced += (len(G0_ava_to_unava_list) - 1)
                                if len(G1_ava_to_unava_list) >= 2:
                                    G1_ava_to_unava_list = [x - matrix_reduced for x in G1_ava_to_unava_list]
                                    traffic_matrix_permutated[[G1_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G1_ava_to_unava_list, :].sum(axis = 0)
                                    rows_to_delete = G1_ava_to_unava_list[1:]
                                    traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                    traffic_matrix_permutated_row_reduced[:,G1_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G1_ava_to_unava_list].sum(axis=1)
                                    cols_to_delete = G1_ava_to_unava_list[1:]
                                    traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                    traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                    matrix_reduced += (len(G1_ava_to_unava_list) - 1)
                                if len(G2_ava_to_unava_list) >= 2:
                                    G2_ava_to_unava_list = [x - matrix_reduced for x in G2_ava_to_unava_list]
                                    traffic_matrix_permutated[[G2_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G2_ava_to_unava_list, :].sum(axis = 0)
                                    rows_to_delete = G2_ava_to_unava_list[1:]
                                    traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                    traffic_matrix_permutated_row_reduced[:,G2_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G2_ava_to_unava_list].sum(axis=1)
                                    cols_to_delete = G2_ava_to_unava_list[1:]
                                    traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                    traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                    matrix_reduced += (len(G2_ava_to_unava_list) - 1)
                                if len(G3_ava_to_unava_list) >= 2:
                                    G3_ava_to_unava_list = [x - matrix_reduced for x in G3_ava_to_unava_list]
                                    traffic_matrix_permutated[[G3_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G3_ava_to_unava_list, :].sum(axis = 0)
                                    rows_to_delete = G3_ava_to_unava_list[1:]
                                    traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                    traffic_matrix_permutated_row_reduced[:,G3_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G3_ava_to_unava_list].sum(axis=1)
                                    cols_to_delete = G3_ava_to_unava_list[1:]
                                    traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                    traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                for i in range(len(group_ava_to_unava)):
                                    if len(group_ava_to_unava[i]) == 0:
                                        zero_row = np.zeros((1, traffic_matrix_permutated.shape[1]), dtype=traffic_matrix_permutated.dtype)
                                        traffic_matrix_permutated_row_sorted = np.vstack((traffic_matrix_permutated[:i, :], zero_row, traffic_matrix_permutated[i:, :]))
                                        zero_col = np.zeros((traffic_matrix_permutated_row_sorted.shape[0], 1), dtype=traffic_matrix_permutated_row_sorted.dtype)
                                        traffic_matrix_permutated_row_col_sorted = np.hstack((traffic_matrix_permutated_row_sorted[:, :i], zero_col, traffic_matrix_permutated_row_sorted[:, i:]))
                                        traffic_matrix_permutated = traffic_matrix_permutated_row_col_sorted
                                
                                current_traffic_matrix = current_traffic_matrix + traffic_matrix_permutated
                                #print(job_queue[0])
                                
                                updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), traffic_matrix_permutated, ava_to_unava)
                                all_jobs[first_num] = (first_num, updated_job_done)

                                t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                                t.start()

                                #print(job_queue[0])
                                job_queue.pop(0)
                            else:
                                updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), -1, ava_to_unava)
                                print("HERE!!")
                                all_jobs[first_num] = (first_num, updated_job_done)

                                t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                                t.start()

                                #print(job_queue[0])
                                job_queue.pop(0)
                            break
                        else:
                            continue
                    continue
            elif (node_count >= first_cpu and first_cpu > 16):
                #print(first_cpu)
                group_idle_counts = []
                for group in range(4):
                    idle_count = 0
                    for node_num in range(group * 16,(group + 1) * 16):
                        if RG.nodes[node_num]["ava"] == "yes":
                            idle_count += 1
                    group_idle_counts.append(idle_count)
                print(f"group_idle_counts: {group_idle_counts}")
                sorted_group_idle_counts = sorted(group_idle_counts, reverse=True)
                print(f"sorted_group_idle_counts: {sorted_group_idle_counts}")
                print(f"group_idle_counts: {group_idle_counts}")
                for i in range(4):
                    if ((i + 1) * sorted_group_idle_counts[i] >= first_cpu):
                        print(first)
                        print(i)
                        one_rack_cpu = first_cpu // (i + 1)
                        one_rack_cpu_arr = [one_rack_cpu for n in range(i + 1)]
                        one_rack_cpu_ex = first_cpu % (i + 1)
                        for j in range(one_rack_cpu_ex):
                            one_rack_cpu_arr[j] = one_rack_cpu_arr[j] + 1
                        print(first_num)
                        print(f"sorted_group_idle_counts: {sorted_group_idle_counts}")
                        print(f"group_idle_counts: {group_idle_counts}")
                        print(f"one_rack_cpu_arr: {one_rack_cpu_arr}")
                        print(f"group_idle_counts: {group_idle_counts}")
                        for n in one_rack_cpu_arr:
                            filtered_counts = [count_ for count_ in group_idle_counts if count_ >= n ]
                            if filtered_counts:
                                min_idle = min(filtered_counts)
                                candidate_groups = [i_ for i_ , cnt in enumerate(group_idle_counts) if cnt == min_idle and cnt >= n]
                                selected_group = candidate_groups[0]
                                idle_nodes = [i_ for i_ in range(selected_group*16, (selected_group+1)*16) if RG.nodes[i_]["ava"] == "yes"]
                                chosen_idle_nodes = [idle_nodes[i] for i in range(int(n))]
                                print(chosen_idle_nodes)
                                for chosen_node in chosen_idle_nodes:
                                    RG.nodes[chosen_node]["ava"] = "no"
                                    nodelist_to_draw.append(chosen_node)
                                    ava_to_unava.append(chosen_node)
                                group_idle_counts[selected_group] = group_idle_counts[selected_group] - n
                                print(group_idle_counts)
                            else:
                                print("エラー")
                                exit()
                                continue
                        # t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava, first,))  # required processing time
                        # t.start()
                        # print(job_queue[0]) 
                        # updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.datetime.now(), -1)
                        # all_jobs[first_num] = (first_num, updated_job_done)
                        # job_queue.pop(0)
                        if first_cpu != 1:
                            traffic_matrix = job_queue[0][1][5]
                            order = np.argsort(ava_to_unava)
                            old_to_new = np.empty_like(order)
                            old_to_new[order] = np.arange(len(ava_to_unava))
                            new_pos = [0] * len(old_to_new)
                            for old_idx, new_idx in enumerate(old_to_new):
                                new_pos[new_idx] = old_idx
                            traffic_matrix_permutated = traffic_matrix[new_pos, :][:, new_pos]
                            print(f"len(ava_to_unava): {len(ava_to_unava)}")
                            print(f"first_cpu: {first_cpu}")
                            print(f"len(traffic_matrix): {len(traffic_matrix)}")
                            new_list = [None] * len(traffic_matrix)
                            for old_idx, new_idx in enumerate(old_to_new):
                                new_list[new_idx] = ava_to_unava[old_idx]
                            print(new_list)
                            G0_ava_to_unava_list = []
                            G1_ava_to_unava_list = []
                            G2_ava_to_unava_list = []
                            G3_ava_to_unava_list = []
                            group_ava_to_unava = [G0_ava_to_unava_list, G1_ava_to_unava_list, G2_ava_to_unava_list, G3_ava_to_unava_list]
                            for i in range(len(new_list)):
                                if new_list[i] <= 15:
                                    G0_ava_to_unava_list.append(i)
                                elif(new_list[i] >= 16) and (new_list[i] <= 31):
                                    G1_ava_to_unava_list.append(i)
                                elif(new_list[i] >= 32) and (new_list[i] <= 47):
                                    G2_ava_to_unava_list.append(i)
                                else:
                                    G3_ava_to_unava_list.append(i)
                            matrix_reduced = 0
                            print(group_ava_to_unava)
                            if len(G0_ava_to_unava_list) >= 2:
                                traffic_matrix_permutated[[G0_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G0_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G0_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G0_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G0_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G0_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G0_ava_to_unava_list) - 1)
                            if len(G1_ava_to_unava_list) >= 2:
                                G1_ava_to_unava_list = [x - matrix_reduced for x in G1_ava_to_unava_list]
                                traffic_matrix_permutated[[G1_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G1_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G1_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G1_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G1_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G1_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G1_ava_to_unava_list) - 1)
                            if len(G2_ava_to_unava_list) >= 2:
                                G2_ava_to_unava_list = [x - matrix_reduced for x in G2_ava_to_unava_list]
                                traffic_matrix_permutated[[G2_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G2_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G2_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G2_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G2_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G2_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G2_ava_to_unava_list) - 1)
                            if len(G3_ava_to_unava_list) >= 2:
                                G3_ava_to_unava_list = [x - matrix_reduced for x in G3_ava_to_unava_list]
                                traffic_matrix_permutated[[G3_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G3_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G3_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G3_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G3_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G3_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            for i in range(len(group_ava_to_unava)):
                                if len(group_ava_to_unava[i]) == 0:
                                    zero_row = np.zeros((1, traffic_matrix_permutated.shape[1]), dtype=traffic_matrix_permutated.dtype)
                                    traffic_matrix_permutated_row_sorted = np.vstack((traffic_matrix_permutated[:i, :], zero_row, traffic_matrix_permutated[i:, :]))
                                    zero_col = np.zeros((traffic_matrix_permutated_row_sorted.shape[0], 1), dtype=traffic_matrix_permutated_row_sorted.dtype)
                                    traffic_matrix_permutated_row_col_sorted = np.hstack((traffic_matrix_permutated_row_sorted[:, :i], zero_col, traffic_matrix_permutated_row_sorted[:, i:]))
                                    traffic_matrix_permutated = traffic_matrix_permutated_row_col_sorted
                            
                            current_traffic_matrix = current_traffic_matrix + traffic_matrix_permutated
                            #print(job_queue[0])
                            
                            updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), traffic_matrix_permutated, ava_to_unava)
                            all_jobs[first_num] = (first_num, updated_job_done)

                            t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                            t.start()

                            #print(job_queue[0])
                            job_queue.pop(0)
                            # print(f"group_idle_counts: {group_idle_counts}")
                        else:
                            updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), -1, ava_to_unava)
                            #print("HERE!!")
                            all_jobs[first_num] = (first_num, updated_job_done)

                            t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                            t.start()

                            #print(job_queue[0])
                            job_queue.pop(0)
                            # print(f"group_idle_counts: {group_idle_counts}")
                        break
                    else:
                        continue
            else:
                continue

            """

            # 消極緩和Local Fat終了

            # 積極緩和Local Fat

            """

            group_idle_counts = []
            flag = True
            # group_large_ava_counts = []
            if (node_count >= first_cpu and first_cpu <= 16):
                for group in range(4):
                    idle_count = 0
                    for node_num in range(group * 16,(group + 1) * 16):
                        if RG.nodes[node_num]["ava"] == "yes":
                            idle_count += 1
                    group_idle_counts.append(idle_count)
                filtered_counts = [count_ for count_ in group_idle_counts if count_ >= first_cpu ]
                sorted_group_idle_counts = sorted(group_idle_counts, reverse = True)
                print("sorted_group_idle_counts")
                print(sorted_group_idle_counts)
                print(f"group_idle_counts: {group_idle_counts}")
                for i in range(1,4):
                    if ((i + 1) * sorted_group_idle_counts[i] >= first_cpu):
                        one_rack_cpu = first_cpu // (i + 1)
                        one_rack_cpu_arr = [one_rack_cpu for n in range(i + 1)]
                        one_rack_cpu_ex = first_cpu % (i + 1)
                        for j in range(one_rack_cpu_ex):
                            one_rack_cpu_arr[j] = one_rack_cpu_arr[j] + 1
                        one_rack_cpu_arr_reversed = sorted(one_rack_cpu_arr, reverse = True)
                        for n in one_rack_cpu_arr_reversed:
                            filtered_counts = [count_ for count_ in group_idle_counts if count_ >= n]
                            if filtered_counts:
                                min_idle = min(filtered_counts)
                                candidate_groups = [i_ for i_ , cnt in enumerate(group_idle_counts) if cnt == min_idle and cnt >= n]
                                selected_group = candidate_groups[0]
                                idle_nodes = [i_ for i_ in range(selected_group*16, (selected_group+1)*16) if RG.nodes[i_]["ava"] == "yes"]
                                chosen_idle_nodes = [idle_nodes[i] for i in range(int(n))]
                                for chosen_node in chosen_idle_nodes:
                                    RG.nodes[chosen_node]["ava"] = "no"
                                    nodelist_to_draw.append(chosen_node)
                                    ava_to_unava.append(chosen_node)
                                print(ava_to_unava)
                                group_idle_counts[selected_group] = group_idle_counts[selected_group] - n
                            else:
                                continue
                        if first_cpu != 1:
                            #print(first_cpu)
                            traffic_matrix = job_queue[0][1][5]
                            # print(traffic_matrix)
                            print(f"job_info: {first_cpu}")
                            print(f"len(traffic_matrix): {len(traffic_matrix)}")
                            print(f"ava_to_unava: {ava_to_unava}")
                            order = np.argsort(ava_to_unava)
                            old_to_new = np.empty_like(order)
                            old_to_new[order] = np.arange(len(ava_to_unava))
                            new_pos = [0] * len(old_to_new)
                            for old_idx, new_idx in enumerate(old_to_new):
                                new_pos[new_idx] = old_idx
                            traffic_matrix_permutated = traffic_matrix[new_pos, :][:, new_pos]
                            new_list = [None] * len(traffic_matrix)
                            for old_idx, new_idx in enumerate(old_to_new):
                                new_list[new_idx] = ava_to_unava[old_idx]
                            print(new_list)
                            G0_ava_to_unava_list = []
                            G1_ava_to_unava_list = []
                            G2_ava_to_unava_list = []
                            G3_ava_to_unava_list = []
                            group_ava_to_unava = [G0_ava_to_unava_list, G1_ava_to_unava_list, G2_ava_to_unava_list, G3_ava_to_unava_list]
                            for i in range(len(new_list)):
                                if new_list[i] <= 15:
                                    G0_ava_to_unava_list.append(i)
                                elif(new_list[i] >= 16) and (new_list[i] <= 31):
                                    G1_ava_to_unava_list.append(i)
                                elif(new_list[i] >= 32) and (new_list[i] <= 47):
                                    G2_ava_to_unava_list.append(i)
                                else:
                                    G3_ava_to_unava_list.append(i)
                            matrix_reduced = 0
                            print(group_ava_to_unava)
                            if len(G0_ava_to_unava_list) >= 2:
                                traffic_matrix_permutated[[G0_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G0_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G0_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G0_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G0_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G0_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G0_ava_to_unava_list) - 1)
                            if len(G1_ava_to_unava_list) >= 2:
                                G1_ava_to_unava_list = [x - matrix_reduced for x in G1_ava_to_unava_list]
                                traffic_matrix_permutated[[G1_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G1_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G1_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G1_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G1_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G1_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G1_ava_to_unava_list) - 1)
                            if len(G2_ava_to_unava_list) >= 2:
                                G2_ava_to_unava_list = [x - matrix_reduced for x in G2_ava_to_unava_list]
                                traffic_matrix_permutated[[G2_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G2_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G2_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G2_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G2_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G2_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G2_ava_to_unava_list) - 1)
                            if len(G3_ava_to_unava_list) >= 2:
                                G3_ava_to_unava_list = [x - matrix_reduced for x in G3_ava_to_unava_list]
                                traffic_matrix_permutated[[G3_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G3_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G3_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G3_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G3_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G3_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            for i in range(len(group_ava_to_unava)):
                                if len(group_ava_to_unava[i]) == 0:
                                    zero_row = np.zeros((1, traffic_matrix_permutated.shape[1]), dtype=traffic_matrix_permutated.dtype)
                                    traffic_matrix_permutated_row_sorted = np.vstack((traffic_matrix_permutated[:i, :], zero_row, traffic_matrix_permutated[i:, :]))
                                    zero_col = np.zeros((traffic_matrix_permutated_row_sorted.shape[0], 1), dtype=traffic_matrix_permutated_row_sorted.dtype)
                                    traffic_matrix_permutated_row_col_sorted = np.hstack((traffic_matrix_permutated_row_sorted[:, :i], zero_col, traffic_matrix_permutated_row_sorted[:, i:]))
                                    traffic_matrix_permutated = traffic_matrix_permutated_row_col_sorted
                            
                            current_traffic_matrix = current_traffic_matrix + traffic_matrix_permutated
                            print(job_queue[0])
                            
                            updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), traffic_matrix_permutated, ava_to_unava)
                            all_jobs[first_num] = (first_num, updated_job_done)

                            t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                            t.start()

                            print(job_queue[0])
                            job_queue.pop(0)
                        else:
                            updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), -1, ava_to_unava)
                            print("HERE!!")
                            all_jobs[first_num] = (first_num, updated_job_done)

                            t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                            t.start()

                            print(job_queue[0])
                            job_queue.pop(0)
                        flag = False
                        break
                    else:
                        continue
                if flag and filtered_counts:
                    min_idle = min(filtered_counts)
                    print(min_idle)
                    print(group_idle_counts)
                    candidate_groups = [i_ for i_ , cnt in enumerate(group_idle_counts) if cnt == min_idle and cnt >= first_cpu]
                    print(candidate_groups)

                    selected_group = candidate_groups[0]
                    print(selected_group)
                    idle_nodes = [i for i in range(selected_group*16, (selected_group+1)*16) if RG.nodes[i]["ava"] == "yes"]
                    chosen_idle_nodes = [idle_nodes[i] for i in range(first_cpu)]
                    for chosen_node in chosen_idle_nodes:
                        RG.nodes[chosen_node]["ava"] = "no"
                        nodelist_to_draw.append(chosen_node)
                        ava_to_unava.append(chosen_node)
                    ava_to_unava_new_list = ava_to_unava

                    if first_cpu != 1:
                        traffic_matrix = job_queue[0][1][5]
                        order = np.argsort(ava_to_unava)
                        old_to_new = np.empty_like(order)
                        old_to_new[order] = np.arange(len(ava_to_unava))
                        new_pos = [0] * len(old_to_new)
                        for old_idx, new_idx in enumerate(old_to_new):
                            new_pos[new_idx] = old_idx
                        traffic_matrix_permutated = traffic_matrix[new_pos, :][:, new_pos]
                        new_list = [None] * len(traffic_matrix)
                        for old_idx, new_idx in enumerate(old_to_new):
                            new_list[new_idx] = ava_to_unava[old_idx]
                        #print(f"ava_to_unava: {ava_to_unava}")
                        #print(f"new_list: {new_list}")
                        G0_ava_to_unava_list = []
                        G1_ava_to_unava_list = []
                        G2_ava_to_unava_list = []
                        G3_ava_to_unava_list = []
                        group_ava_to_unava = [G0_ava_to_unava_list, G1_ava_to_unava_list, G2_ava_to_unava_list, G3_ava_to_unava_list]
                        for i in range(len(new_list)):
                            if new_list[i] <= 15:
                                G0_ava_to_unava_list.append(i)
                            elif(new_list[i] >= 16) and (new_list[i] <= 31):
                                G1_ava_to_unava_list.append(i)
                            elif(new_list[i] >= 32) and (new_list[i] <= 47):
                                G2_ava_to_unava_list.append(i)
                            else:
                                G3_ava_to_unava_list.append(i)
                        matrix_reduced = 0
                        #print(group_ava_to_unava)
                        if len(G0_ava_to_unava_list) >= 2:
                            traffic_matrix_permutated[[G0_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G0_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G0_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G0_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G0_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G0_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            matrix_reduced += (len(G0_ava_to_unava_list) - 1)
                        if len(G1_ava_to_unava_list) >= 2:
                            G1_ava_to_unava_list = [x - matrix_reduced for x in G1_ava_to_unava_list]
                            traffic_matrix_permutated[[G1_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G1_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G1_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G1_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G1_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G1_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            matrix_reduced += (len(G1_ava_to_unava_list) - 1)
                        if len(G2_ava_to_unava_list) >= 2:
                            G2_ava_to_unava_list = [x - matrix_reduced for x in G2_ava_to_unava_list]
                            traffic_matrix_permutated[[G2_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G2_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G2_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G2_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G2_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G2_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            matrix_reduced += (len(G2_ava_to_unava_list) - 1)
                        if len(G3_ava_to_unava_list) >= 2:
                            G3_ava_to_unava_list = [x - matrix_reduced for x in G3_ava_to_unava_list]
                            traffic_matrix_permutated[[G3_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G3_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G3_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G3_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G3_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G3_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                        for i in range(len(group_ava_to_unava)):
                            if len(group_ava_to_unava[i]) == 0:
                                zero_row = np.zeros((1, traffic_matrix_permutated.shape[1]), dtype=traffic_matrix_permutated.dtype)
                                traffic_matrix_permutated_row_sorted = np.vstack((traffic_matrix_permutated[:i, :], zero_row, traffic_matrix_permutated[i:, :]))
                                zero_col = np.zeros((traffic_matrix_permutated_row_sorted.shape[0], 1), dtype=traffic_matrix_permutated_row_sorted.dtype)
                                traffic_matrix_permutated_row_col_sorted = np.hstack((traffic_matrix_permutated_row_sorted[:, :i], zero_col, traffic_matrix_permutated_row_sorted[:, i:]))
                                traffic_matrix_permutated = traffic_matrix_permutated_row_col_sorted
                        
                        current_traffic_matrix = current_traffic_matrix + traffic_matrix_permutated
                        #print(job_queue[0])
                        
                        updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), traffic_matrix_permutated, ava_to_unava)
                        all_jobs[first_num] = (first_num, updated_job_done)

                        t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                        t.start()

                        #print(job_queue[0])
                        job_queue.pop(0)
                        print(f"group_idle_counts: {group_idle_counts}")
                    else:
                        updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), -1, ava_to_unava)
                        #print("HERE!!")
                        all_jobs[first_num] = (first_num, updated_job_done)

                        t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                        t.start()

                        #print(job_queue[0])
                        job_queue.pop(0)
                        print(f"group_idle_counts: {group_idle_counts}")

                    # t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava, first,))  # required processing time
                    # t.start()
                    # updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.datetime.now(), -1)
                    # all_jobs[first_num] = (first_num, updated_job_done)
                    # job_queue.pop(0)
                    flag = True
                    continue

                continue
            elif (node_count >= first_cpu and first_cpu > 16):
                #print(first_cpu)
                group_idle_counts = []
                for group in range(4):
                    idle_count = 0
                    for node_num in range(group * 16,(group + 1) * 16):
                        if RG.nodes[node_num]["ava"] == "yes":
                            idle_count += 1
                    group_idle_counts.append(idle_count)
                print(f"group_idle_counts: {group_idle_counts}")
                sorted_group_idle_counts = sorted(group_idle_counts, reverse=True)
                print(f"sorted_group_idle_counts: {sorted_group_idle_counts}")
                print(f"group_idle_counts: {group_idle_counts}")
                use_group_num = first_cpu // 16
                if first_cpu % 16 != 0:
                    use_group_num += 1
                if use_group_num >= 3:
                    use_group_num = 2
                for i in range((use_group_num + 1), 4):
                    if ((i + 1) * sorted_group_idle_counts[i] >= first_cpu):
                        print(first)
                        print(i)
                        one_rack_cpu = first_cpu // (i + 1)
                        one_rack_cpu_arr = [one_rack_cpu for n in range(i + 1)]
                        one_rack_cpu_ex = first_cpu % (i + 1)
                        for j in range(one_rack_cpu_ex):
                            one_rack_cpu_arr[j] = one_rack_cpu_arr[j] + 1
                        print(first_num)
                        print(f"sorted_group_idle_counts: {sorted_group_idle_counts}")
                        print(f"group_idle_counts: {group_idle_counts}")
                        print(f"one_rack_cpu_arr: {one_rack_cpu_arr}")
                        print(f"group_idle_counts: {group_idle_counts}")
                        for n in one_rack_cpu_arr:
                            filtered_counts = [count_ for count_ in group_idle_counts if count_ >= n ]
                            if filtered_counts:
                                min_idle = min(filtered_counts)
                                candidate_groups = [i_ for i_ , cnt in enumerate(group_idle_counts) if cnt == min_idle and cnt >= n]
                                selected_group = candidate_groups[0]
                                idle_nodes = [i_ for i_ in range(selected_group*16, (selected_group+1)*16) if RG.nodes[i_]["ava"] == "yes"]
                                chosen_idle_nodes = [idle_nodes[i] for i in range(int(n))]
                                print(chosen_idle_nodes)
                                for chosen_node in chosen_idle_nodes:
                                    RG.nodes[chosen_node]["ava"] = "no"
                                    nodelist_to_draw.append(chosen_node)
                                    ava_to_unava.append(chosen_node)
                                group_idle_counts[selected_group] = group_idle_counts[selected_group] - n
                                print(group_idle_counts)
                            else:
                                print("エラー")
                                exit()
                                continue
                        # t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava, first,))  # required processing time
                        # t.start()
                        # print(job_queue[0]) 
                        # updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.datetime.now(), -1)
                        # all_jobs[first_num] = (first_num, updated_job_done)
                        # job_queue.pop(0)
                        if first_cpu != 1:
                            traffic_matrix = job_queue[0][1][5]
                            order = np.argsort(ava_to_unava)
                            old_to_new = np.empty_like(order)
                            old_to_new[order] = np.arange(len(ava_to_unava))
                            new_pos = [0] * len(old_to_new)
                            for old_idx, new_idx in enumerate(old_to_new):
                                new_pos[new_idx] = old_idx
                            traffic_matrix_permutated = traffic_matrix[new_pos, :][:, new_pos]
                            print(f"len(ava_to_unava): {len(ava_to_unava)}")
                            print(f"first_cpu: {first_cpu}")
                            print(f"len(traffic_matrix): {len(traffic_matrix)}")
                            new_list = [None] * len(traffic_matrix)
                            for old_idx, new_idx in enumerate(old_to_new):
                                new_list[new_idx] = ava_to_unava[old_idx]
                            print(new_list)
                            G0_ava_to_unava_list = []
                            G1_ava_to_unava_list = []
                            G2_ava_to_unava_list = []
                            G3_ava_to_unava_list = []
                            group_ava_to_unava = [G0_ava_to_unava_list, G1_ava_to_unava_list, G2_ava_to_unava_list, G3_ava_to_unava_list]
                            for i in range(len(new_list)):
                                if new_list[i] <= 15:
                                    G0_ava_to_unava_list.append(i)
                                elif(new_list[i] >= 16) and (new_list[i] <= 31):
                                    G1_ava_to_unava_list.append(i)
                                elif(new_list[i] >= 32) and (new_list[i] <= 47):
                                    G2_ava_to_unava_list.append(i)
                                else:
                                    G3_ava_to_unava_list.append(i)
                            matrix_reduced = 0
                            print(group_ava_to_unava)
                            if len(G0_ava_to_unava_list) >= 2:
                                traffic_matrix_permutated[[G0_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G0_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G0_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G0_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G0_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G0_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G0_ava_to_unava_list) - 1)
                            if len(G1_ava_to_unava_list) >= 2:
                                G1_ava_to_unava_list = [x - matrix_reduced for x in G1_ava_to_unava_list]
                                traffic_matrix_permutated[[G1_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G1_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G1_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G1_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G1_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G1_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G1_ava_to_unava_list) - 1)
                            if len(G2_ava_to_unava_list) >= 2:
                                G2_ava_to_unava_list = [x - matrix_reduced for x in G2_ava_to_unava_list]
                                traffic_matrix_permutated[[G2_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G2_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G2_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G2_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G2_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G2_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G2_ava_to_unava_list) - 1)
                            if len(G3_ava_to_unava_list) >= 2:
                                G3_ava_to_unava_list = [x - matrix_reduced for x in G3_ava_to_unava_list]
                                traffic_matrix_permutated[[G3_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G3_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G3_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G3_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G3_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G3_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            for i in range(len(group_ava_to_unava)):
                                if len(group_ava_to_unava[i]) == 0:
                                    zero_row = np.zeros((1, traffic_matrix_permutated.shape[1]), dtype=traffic_matrix_permutated.dtype)
                                    traffic_matrix_permutated_row_sorted = np.vstack((traffic_matrix_permutated[:i, :], zero_row, traffic_matrix_permutated[i:, :]))
                                    zero_col = np.zeros((traffic_matrix_permutated_row_sorted.shape[0], 1), dtype=traffic_matrix_permutated_row_sorted.dtype)
                                    traffic_matrix_permutated_row_col_sorted = np.hstack((traffic_matrix_permutated_row_sorted[:, :i], zero_col, traffic_matrix_permutated_row_sorted[:, i:]))
                                    traffic_matrix_permutated = traffic_matrix_permutated_row_col_sorted
                            
                            current_traffic_matrix = current_traffic_matrix + traffic_matrix_permutated
                            #print(job_queue[0])
                            
                            updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), traffic_matrix_permutated, ava_to_unava)
                            all_jobs[first_num] = (first_num, updated_job_done)

                            t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                            t.start()

                            #print(job_queue[0])
                            job_queue.pop(0)
                            # print(f"group_idle_counts: {group_idle_counts}")
                        else:
                            updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), -1, ava_to_unava)
                            #print("HERE!!")
                            all_jobs[first_num] = (first_num, updated_job_done)

                            t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                            t.start()

                            #print(job_queue[0])
                            job_queue.pop(0)
                            # print(f"group_idle_counts: {group_idle_counts}")
                        break
                    else:
                        continue
            else:
                continue
            
            """

            # 積極緩和Local Fat終了

            # 積極緩和平滑化Local Fat

            """

            group_idle_counts = []
            flag = True
            # group_large_ava_counts = []
            if (node_count >= first_cpu and first_cpu <= 16):
                for group in range(4):
                    idle_count = 0
                    for node_num in range(group * 16,(group + 1) * 16):
                        if RG.nodes[node_num]["ava"] == "yes":
                            idle_count += 1
                    group_idle_counts.append(idle_count)
                filtered_counts = [count_ for count_ in group_idle_counts if count_ >= first_cpu ]
                sorted_group_idle_counts = sorted(group_idle_counts, reverse = True)
                print("sorted_group_idle_counts")
                print(sorted_group_idle_counts)
                print(f"group_idle_counts: {group_idle_counts}")
                for i in range(1,4):
                    if ((i + 1) * sorted_group_idle_counts[i] >= first_cpu):
                        one_rack_cpu = first_cpu // (i + 1)
                        one_rack_cpu_arr = [one_rack_cpu for n in range(i + 1)]
                        one_rack_cpu_ex = first_cpu % (i + 1)
                        for j in range(one_rack_cpu_ex):
                            one_rack_cpu_arr[j] = one_rack_cpu_arr[j] + 1
                        one_rack_cpu_arr_reversed = sorted(one_rack_cpu_arr, reverse = True)
                        for n in one_rack_cpu_arr_reversed:
                            filtered_counts = [count_ for count_ in group_idle_counts if count_ >= n]
                            if filtered_counts:
                                min_idle = max(filtered_counts)
                                candidate_groups = [i_ for i_ , cnt in enumerate(group_idle_counts) if cnt == min_idle and cnt >= n]
                                selected_group = candidate_groups[0]
                                idle_nodes = [i_ for i_ in range(selected_group*16, (selected_group+1)*16) if RG.nodes[i_]["ava"] == "yes"]
                                chosen_idle_nodes = [idle_nodes[i] for i in range(int(n))]
                                for chosen_node in chosen_idle_nodes:
                                    RG.nodes[chosen_node]["ava"] = "no"
                                    nodelist_to_draw.append(chosen_node)
                                    ava_to_unava.append(chosen_node)
                                print(ava_to_unava)
                                group_idle_counts[selected_group] = group_idle_counts[selected_group] - n
                            else:
                                continue
                        if first_cpu != 1:
                            #print(first_cpu)
                            traffic_matrix = job_queue[0][1][5]
                            # print(traffic_matrix)
                            print(f"job_info: {first_cpu}")
                            print(f"len(traffic_matrix): {len(traffic_matrix)}")
                            print(f"ava_to_unava: {ava_to_unava}")
                            order = np.argsort(ava_to_unava)
                            old_to_new = np.empty_like(order)
                            old_to_new[order] = np.arange(len(ava_to_unava))
                            new_pos = [0] * len(old_to_new)
                            for old_idx, new_idx in enumerate(old_to_new):
                                new_pos[new_idx] = old_idx
                            traffic_matrix_permutated = traffic_matrix[new_pos, :][:, new_pos]
                            new_list = [None] * len(traffic_matrix)
                            for old_idx, new_idx in enumerate(old_to_new):
                                new_list[new_idx] = ava_to_unava[old_idx]
                            print(new_list)
                            G0_ava_to_unava_list = []
                            G1_ava_to_unava_list = []
                            G2_ava_to_unava_list = []
                            G3_ava_to_unava_list = []
                            group_ava_to_unava = [G0_ava_to_unava_list, G1_ava_to_unava_list, G2_ava_to_unava_list, G3_ava_to_unava_list]
                            for i in range(len(new_list)):
                                if new_list[i] <= 15:
                                    G0_ava_to_unava_list.append(i)
                                elif(new_list[i] >= 16) and (new_list[i] <= 31):
                                    G1_ava_to_unava_list.append(i)
                                elif(new_list[i] >= 32) and (new_list[i] <= 47):
                                    G2_ava_to_unava_list.append(i)
                                else:
                                    G3_ava_to_unava_list.append(i)
                            matrix_reduced = 0
                            print(group_ava_to_unava)
                            if len(G0_ava_to_unava_list) >= 2:
                                traffic_matrix_permutated[[G0_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G0_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G0_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G0_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G0_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G0_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G0_ava_to_unava_list) - 1)
                            if len(G1_ava_to_unava_list) >= 2:
                                G1_ava_to_unava_list = [x - matrix_reduced for x in G1_ava_to_unava_list]
                                traffic_matrix_permutated[[G1_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G1_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G1_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G1_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G1_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G1_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G1_ava_to_unava_list) - 1)
                            if len(G2_ava_to_unava_list) >= 2:
                                G2_ava_to_unava_list = [x - matrix_reduced for x in G2_ava_to_unava_list]
                                traffic_matrix_permutated[[G2_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G2_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G2_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G2_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G2_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G2_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G2_ava_to_unava_list) - 1)
                            if len(G3_ava_to_unava_list) >= 2:
                                G3_ava_to_unava_list = [x - matrix_reduced for x in G3_ava_to_unava_list]
                                traffic_matrix_permutated[[G3_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G3_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G3_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G3_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G3_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G3_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            for i in range(len(group_ava_to_unava)):
                                if len(group_ava_to_unava[i]) == 0:
                                    zero_row = np.zeros((1, traffic_matrix_permutated.shape[1]), dtype=traffic_matrix_permutated.dtype)
                                    traffic_matrix_permutated_row_sorted = np.vstack((traffic_matrix_permutated[:i, :], zero_row, traffic_matrix_permutated[i:, :]))
                                    zero_col = np.zeros((traffic_matrix_permutated_row_sorted.shape[0], 1), dtype=traffic_matrix_permutated_row_sorted.dtype)
                                    traffic_matrix_permutated_row_col_sorted = np.hstack((traffic_matrix_permutated_row_sorted[:, :i], zero_col, traffic_matrix_permutated_row_sorted[:, i:]))
                                    traffic_matrix_permutated = traffic_matrix_permutated_row_col_sorted
                            
                            current_traffic_matrix = current_traffic_matrix + traffic_matrix_permutated
                            print(job_queue[0])
                            
                            updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), traffic_matrix_permutated, ava_to_unava)
                            all_jobs[first_num] = (first_num, updated_job_done)

                            t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                            t.start()

                            print(job_queue[0])
                            job_queue.pop(0)
                        else:
                            updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), -1, ava_to_unava)
                            print("HERE!!")
                            all_jobs[first_num] = (first_num, updated_job_done)

                            t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                            t.start()

                            print(job_queue[0])
                            job_queue.pop(0)
                        flag = False
                        break
                    else:
                        continue
                if flag and filtered_counts:
                    min_idle = max(filtered_counts)
                    print(min_idle)
                    print(group_idle_counts)
                    candidate_groups = [i_ for i_ , cnt in enumerate(group_idle_counts) if cnt == min_idle and cnt >= first_cpu]
                    print(candidate_groups)

                    selected_group = candidate_groups[0]
                    print(selected_group)
                    idle_nodes = [i for i in range(selected_group*16, (selected_group+1)*16) if RG.nodes[i]["ava"] == "yes"]
                    chosen_idle_nodes = [idle_nodes[i] for i in range(first_cpu)]
                    for chosen_node in chosen_idle_nodes:
                        RG.nodes[chosen_node]["ava"] = "no"
                        nodelist_to_draw.append(chosen_node)
                        ava_to_unava.append(chosen_node)
                    ava_to_unava_new_list = ava_to_unava

                    if first_cpu != 1:
                        traffic_matrix = job_queue[0][1][5]
                        order = np.argsort(ava_to_unava)
                        old_to_new = np.empty_like(order)
                        old_to_new[order] = np.arange(len(ava_to_unava))
                        new_pos = [0] * len(old_to_new)
                        for old_idx, new_idx in enumerate(old_to_new):
                            new_pos[new_idx] = old_idx
                        traffic_matrix_permutated = traffic_matrix[new_pos, :][:, new_pos]
                        new_list = [None] * len(traffic_matrix)
                        for old_idx, new_idx in enumerate(old_to_new):
                            new_list[new_idx] = ava_to_unava[old_idx]
                        #print(f"ava_to_unava: {ava_to_unava}")
                        #print(f"new_list: {new_list}")
                        G0_ava_to_unava_list = []
                        G1_ava_to_unava_list = []
                        G2_ava_to_unava_list = []
                        G3_ava_to_unava_list = []
                        group_ava_to_unava = [G0_ava_to_unava_list, G1_ava_to_unava_list, G2_ava_to_unava_list, G3_ava_to_unava_list]
                        for i in range(len(new_list)):
                            if new_list[i] <= 15:
                                G0_ava_to_unava_list.append(i)
                            elif(new_list[i] >= 16) and (new_list[i] <= 31):
                                G1_ava_to_unava_list.append(i)
                            elif(new_list[i] >= 32) and (new_list[i] <= 47):
                                G2_ava_to_unava_list.append(i)
                            else:
                                G3_ava_to_unava_list.append(i)
                        matrix_reduced = 0
                        #print(group_ava_to_unava)
                        if len(G0_ava_to_unava_list) >= 2:
                            traffic_matrix_permutated[[G0_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G0_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G0_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G0_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G0_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G0_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            matrix_reduced += (len(G0_ava_to_unava_list) - 1)
                        if len(G1_ava_to_unava_list) >= 2:
                            G1_ava_to_unava_list = [x - matrix_reduced for x in G1_ava_to_unava_list]
                            traffic_matrix_permutated[[G1_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G1_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G1_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G1_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G1_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G1_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            matrix_reduced += (len(G1_ava_to_unava_list) - 1)
                        if len(G2_ava_to_unava_list) >= 2:
                            G2_ava_to_unava_list = [x - matrix_reduced for x in G2_ava_to_unava_list]
                            traffic_matrix_permutated[[G2_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G2_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G2_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G2_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G2_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G2_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            matrix_reduced += (len(G2_ava_to_unava_list) - 1)
                        if len(G3_ava_to_unava_list) >= 2:
                            G3_ava_to_unava_list = [x - matrix_reduced for x in G3_ava_to_unava_list]
                            traffic_matrix_permutated[[G3_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G3_ava_to_unava_list, :].sum(axis = 0)
                            rows_to_delete = G3_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                            traffic_matrix_permutated_row_reduced[:,G3_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G3_ava_to_unava_list].sum(axis=1)
                            cols_to_delete = G3_ava_to_unava_list[1:]
                            traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                            traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                        for i in range(len(group_ava_to_unava)):
                            if len(group_ava_to_unava[i]) == 0:
                                zero_row = np.zeros((1, traffic_matrix_permutated.shape[1]), dtype=traffic_matrix_permutated.dtype)
                                traffic_matrix_permutated_row_sorted = np.vstack((traffic_matrix_permutated[:i, :], zero_row, traffic_matrix_permutated[i:, :]))
                                zero_col = np.zeros((traffic_matrix_permutated_row_sorted.shape[0], 1), dtype=traffic_matrix_permutated_row_sorted.dtype)
                                traffic_matrix_permutated_row_col_sorted = np.hstack((traffic_matrix_permutated_row_sorted[:, :i], zero_col, traffic_matrix_permutated_row_sorted[:, i:]))
                                traffic_matrix_permutated = traffic_matrix_permutated_row_col_sorted
                        
                        current_traffic_matrix = current_traffic_matrix + traffic_matrix_permutated
                        #print(job_queue[0])
                        
                        updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), traffic_matrix_permutated, ava_to_unava)
                        all_jobs[first_num] = (first_num, updated_job_done)

                        t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                        t.start()

                        #print(job_queue[0])
                        job_queue.pop(0)
                        print(f"group_idle_counts: {group_idle_counts}")
                    else:
                        updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), -1, ava_to_unava)
                        #print("HERE!!")
                        all_jobs[first_num] = (first_num, updated_job_done)

                        t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                        t.start()

                        #print(job_queue[0])
                        job_queue.pop(0)
                        print(f"group_idle_counts: {group_idle_counts}")

                    # t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava, first,))  # required processing time
                    # t.start()
                    # updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.datetime.now(), -1)
                    # all_jobs[first_num] = (first_num, updated_job_done)
                    # job_queue.pop(0)
                    flag = True
                    continue

                continue
            elif (node_count >= first_cpu and first_cpu > 16):
                #print(first_cpu)
                group_idle_counts = []
                for group in range(4):
                    idle_count = 0
                    for node_num in range(group * 16,(group + 1) * 16):
                        if RG.nodes[node_num]["ava"] == "yes":
                            idle_count += 1
                    group_idle_counts.append(idle_count)
                print(f"group_idle_counts: {group_idle_counts}")
                sorted_group_idle_counts = sorted(group_idle_counts, reverse=True)
                print(f"sorted_group_idle_counts: {sorted_group_idle_counts}")
                print(f"group_idle_counts: {group_idle_counts}")
                use_group_num = first_cpu // 16
                if first_cpu % 16 != 0:
                    use_group_num += 1
                if use_group_num >= 3:
                    use_group_num = 2
                for i in range((use_group_num + 1), 4):
                    if ((i + 1) * sorted_group_idle_counts[i] >= first_cpu):
                        print(first)
                        print(i)
                        one_rack_cpu = first_cpu // (i + 1)
                        one_rack_cpu_arr = [one_rack_cpu for n in range(i + 1)]
                        one_rack_cpu_ex = first_cpu % (i + 1)
                        for j in range(one_rack_cpu_ex):
                            one_rack_cpu_arr[j] = one_rack_cpu_arr[j] + 1
                        print(first_num)
                        print(f"sorted_group_idle_counts: {sorted_group_idle_counts}")
                        print(f"group_idle_counts: {group_idle_counts}")
                        print(f"one_rack_cpu_arr: {one_rack_cpu_arr}")
                        print(f"group_idle_counts: {group_idle_counts}")
                        for n in one_rack_cpu_arr:
                            filtered_counts = [count_ for count_ in group_idle_counts if count_ >= n ]
                            if filtered_counts:
                                min_idle = min(filtered_counts)
                                candidate_groups = [i_ for i_ , cnt in enumerate(group_idle_counts) if cnt == min_idle and cnt >= n]
                                selected_group = candidate_groups[0]
                                idle_nodes = [i_ for i_ in range(selected_group*16, (selected_group+1)*16) if RG.nodes[i_]["ava"] == "yes"]
                                chosen_idle_nodes = [idle_nodes[i] for i in range(int(n))]
                                print(chosen_idle_nodes)
                                for chosen_node in chosen_idle_nodes:
                                    RG.nodes[chosen_node]["ava"] = "no"
                                    nodelist_to_draw.append(chosen_node)
                                    ava_to_unava.append(chosen_node)
                                group_idle_counts[selected_group] = group_idle_counts[selected_group] - n
                                print(group_idle_counts)
                            else:
                                print("エラー")
                                exit()
                                continue
                        # t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava, first,))  # required processing time
                        # t.start()
                        # print(job_queue[0]) 
                        # updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.datetime.now(), -1)
                        # all_jobs[first_num] = (first_num, updated_job_done)
                        # job_queue.pop(0)
                        if first_cpu != 1:
                            traffic_matrix = job_queue[0][1][5]
                            order = np.argsort(ava_to_unava)
                            old_to_new = np.empty_like(order)
                            old_to_new[order] = np.arange(len(ava_to_unava))
                            new_pos = [0] * len(old_to_new)
                            for old_idx, new_idx in enumerate(old_to_new):
                                new_pos[new_idx] = old_idx
                            traffic_matrix_permutated = traffic_matrix[new_pos, :][:, new_pos]
                            print(f"len(ava_to_unava): {len(ava_to_unava)}")
                            print(f"first_cpu: {first_cpu}")
                            print(f"len(traffic_matrix): {len(traffic_matrix)}")
                            new_list = [None] * len(traffic_matrix)
                            for old_idx, new_idx in enumerate(old_to_new):
                                new_list[new_idx] = ava_to_unava[old_idx]
                            print(new_list)
                            G0_ava_to_unava_list = []
                            G1_ava_to_unava_list = []
                            G2_ava_to_unava_list = []
                            G3_ava_to_unava_list = []
                            group_ava_to_unava = [G0_ava_to_unava_list, G1_ava_to_unava_list, G2_ava_to_unava_list, G3_ava_to_unava_list]
                            for i in range(len(new_list)):
                                if new_list[i] <= 15:
                                    G0_ava_to_unava_list.append(i)
                                elif(new_list[i] >= 16) and (new_list[i] <= 31):
                                    G1_ava_to_unava_list.append(i)
                                elif(new_list[i] >= 32) and (new_list[i] <= 47):
                                    G2_ava_to_unava_list.append(i)
                                else:
                                    G3_ava_to_unava_list.append(i)
                            matrix_reduced = 0
                            print(group_ava_to_unava)
                            if len(G0_ava_to_unava_list) >= 2:
                                traffic_matrix_permutated[[G0_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G0_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G0_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G0_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G0_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G0_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G0_ava_to_unava_list) - 1)
                            if len(G1_ava_to_unava_list) >= 2:
                                G1_ava_to_unava_list = [x - matrix_reduced for x in G1_ava_to_unava_list]
                                traffic_matrix_permutated[[G1_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G1_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G1_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G1_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G1_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G1_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G1_ava_to_unava_list) - 1)
                            if len(G2_ava_to_unava_list) >= 2:
                                G2_ava_to_unava_list = [x - matrix_reduced for x in G2_ava_to_unava_list]
                                traffic_matrix_permutated[[G2_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G2_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G2_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G2_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G2_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G2_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                                matrix_reduced += (len(G2_ava_to_unava_list) - 1)
                            if len(G3_ava_to_unava_list) >= 2:
                                G3_ava_to_unava_list = [x - matrix_reduced for x in G3_ava_to_unava_list]
                                traffic_matrix_permutated[[G3_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G3_ava_to_unava_list, :].sum(axis = 0)
                                rows_to_delete = G3_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                                traffic_matrix_permutated_row_reduced[:,G3_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G3_ava_to_unava_list].sum(axis=1)
                                cols_to_delete = G3_ava_to_unava_list[1:]
                                traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                                traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                            for i in range(len(group_ava_to_unava)):
                                if len(group_ava_to_unava[i]) == 0:
                                    zero_row = np.zeros((1, traffic_matrix_permutated.shape[1]), dtype=traffic_matrix_permutated.dtype)
                                    traffic_matrix_permutated_row_sorted = np.vstack((traffic_matrix_permutated[:i, :], zero_row, traffic_matrix_permutated[i:, :]))
                                    zero_col = np.zeros((traffic_matrix_permutated_row_sorted.shape[0], 1), dtype=traffic_matrix_permutated_row_sorted.dtype)
                                    traffic_matrix_permutated_row_col_sorted = np.hstack((traffic_matrix_permutated_row_sorted[:, :i], zero_col, traffic_matrix_permutated_row_sorted[:, i:]))
                                    traffic_matrix_permutated = traffic_matrix_permutated_row_col_sorted
                            
                            current_traffic_matrix = current_traffic_matrix + traffic_matrix_permutated
                            #print(job_queue[0])
                            
                            updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), traffic_matrix_permutated, ava_to_unava)
                            all_jobs[first_num] = (first_num, updated_job_done)

                            t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                            t.start()

                            #print(job_queue[0])
                            job_queue.pop(0)
                            # print(f"group_idle_counts: {group_idle_counts}")
                        else:
                            updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), -1, ava_to_unava)
                            #print("HERE!!")
                            all_jobs[first_num] = (first_num, updated_job_done)

                            t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))  # required processing time
                            t.start()

                            #print(job_queue[0])
                            job_queue.pop(0)
                            # print(f"group_idle_counts: {group_idle_counts}")
                        break
                    else:
                        continue
            else:
                continue

            """

            # 積極緩和平滑化Local Fat終了

            # OCSアルゴリズム最大強化(自由FullFat)

            """
            
            おいてあるところに繋ぐ
            > 空いていて置ける中でランダムに置く
            > トラフィックを発生させ、OCS再構成アルゴリズムを行使
            
            """
            
            
            
            if (node_count >= first_cpu):
                random_0_63_list = list(range(64))
                random.shuffle(random_0_63_list)
                for i in range(first_cpu):
                    for j in random_0_63_list:
                        if RG.nodes[j]["ava"] == "yes":
                            RG.nodes[j]["ava"] = "no"
                            nodelist_to_draw.append(j)
                            ava_to_unava.append(j)
                            break
                        else:
                            continue
                print(first_num, ava_to_unava)
                ava_to_unava_new_list = ava_to_unava
                print("first_cpu =", first_cpu, type(first_cpu))
                #traffic_matrix = np.zeros((first_cpu, first_cpu), dtype=float) # トラフィック行列をここにぶち込む
                if first_cpu != 1:
                    traffic_matrix = job_queue[0][1][5]
                    traffic_matrix_64_64 = np.zeros((64, 64))
                    for i in range(len(ava_to_unava)):
                        for j in range(len(ava_to_unava)):
                            traffic_matrix_64_64[ava_to_unava[i]][ava_to_unava[j]] += traffic_matrix[i][j]
                    # traffic_matrix = job_queue[0][1][5]
                    order = np.argsort(ava_to_unava)
                    old_to_new = np.empty_like(order)
                    old_to_new[order] = np.arange(len(ava_to_unava))
                    new_pos = [0] * len(old_to_new)
                    for old_idx, new_idx in enumerate(old_to_new):
                        new_pos[new_idx] = old_idx
                    traffic_matrix_permutated = traffic_matrix[new_pos, :][:, new_pos]
                    new_list = [None] * len(traffic_matrix)
                    for old_idx, new_idx in enumerate(old_to_new):
                        new_list[new_idx] = ava_to_unava[old_idx]
                    print(new_list)
                    G0_ava_to_unava_list = []
                    G1_ava_to_unava_list = []
                    G2_ava_to_unava_list = []
                    G3_ava_to_unava_list = []
                    group_ava_to_unava = [G0_ava_to_unava_list, G1_ava_to_unava_list, G2_ava_to_unava_list, G3_ava_to_unava_list]
                    for i in range(len(new_list)):
                        if new_list[i] <= 15:
                            G0_ava_to_unava_list.append(i)
                        elif(new_list[i] >= 16) and (new_list[i] <= 31):
                            G1_ava_to_unava_list.append(i)
                        elif(new_list[i] >= 32) and (new_list[i] <= 47):
                            G2_ava_to_unava_list.append(i)
                        else:
                            G3_ava_to_unava_list.append(i)
                    matrix_reduced = 0
                    print(group_ava_to_unava)
                    if len(G0_ava_to_unava_list) >= 2:
                        traffic_matrix_permutated[[G0_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G0_ava_to_unava_list, :].sum(axis = 0)
                        rows_to_delete = G0_ava_to_unava_list[1:]
                        traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                        traffic_matrix_permutated_row_reduced[:,G0_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G0_ava_to_unava_list].sum(axis=1)
                        cols_to_delete = G0_ava_to_unava_list[1:]
                        traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                        traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                        matrix_reduced += (len(G0_ava_to_unava_list) - 1)
                        
                    if len(G1_ava_to_unava_list) >= 2:
                        G1_ava_to_unava_list = [x - matrix_reduced for x in G1_ava_to_unava_list]
                        traffic_matrix_permutated[[G1_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G1_ava_to_unava_list, :].sum(axis = 0)
                        rows_to_delete = G1_ava_to_unava_list[1:]
                        traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                        traffic_matrix_permutated_row_reduced[:,G1_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G1_ava_to_unava_list].sum(axis=1)
                        cols_to_delete = G1_ava_to_unava_list[1:]
                        traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                        traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                        G1_ava_to_unava_list_new = [0] * len(G2_ava_to_unava_list)
                        matrix_reduced += (len(G1_ava_to_unava_list) - 1)
                        # G2_ava_to_unava_list = [x - matrix_reduced for x in G2_ava_to_unava_list]
                        # G3_ava_to_unava_list = [x - matrix_reduced for x in G3_ava_to_unava_list]
                    if len(G2_ava_to_unava_list) >= 2:
                        G2_ava_to_unava_list = [x - matrix_reduced for x in G2_ava_to_unava_list]
                        traffic_matrix_permutated[[G2_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G2_ava_to_unava_list, :].sum(axis = 0)
                        rows_to_delete = G2_ava_to_unava_list[1:]
                        traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                        traffic_matrix_permutated_row_reduced[:,G2_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G2_ava_to_unava_list].sum(axis=1)
                        cols_to_delete = G2_ava_to_unava_list[1:]
                        traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                        traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                        G1_ava_to_unava_list_new = [0] * len(G3_ava_to_unava_list)
                        matrix_reduced += (len(G2_ava_to_unava_list) - 1)
                        # G3_ava_to_unava_list = [x - matrix_reduced for x in G3_ava_to_unava_list]
                    if len(G3_ava_to_unava_list) >= 2:
                        G3_ava_to_unava_list = [x - matrix_reduced for x in G3_ava_to_unava_list]
                        traffic_matrix_permutated[[G3_ava_to_unava_list[0]],:] = traffic_matrix_permutated[G3_ava_to_unava_list, :].sum(axis = 0)
                        rows_to_delete = G3_ava_to_unava_list[1:]
                        traffic_matrix_permutated_row_reduced = np.delete(traffic_matrix_permutated, rows_to_delete, axis=0)
                        traffic_matrix_permutated_row_reduced[:,G3_ava_to_unava_list[0]] = traffic_matrix_permutated_row_reduced[:, G3_ava_to_unava_list].sum(axis=1)
                        cols_to_delete = G3_ava_to_unava_list[1:]
                        traffic_matrix_permutated_row_cols_reduced = np.delete(traffic_matrix_permutated_row_reduced, cols_to_delete, axis=1)
                        traffic_matrix_permutated = traffic_matrix_permutated_row_cols_reduced
                        # matrix_reduced += (len(G3_ava_to_unava_list) - 1)
                    #print("G3 after")
                    #print(G3_ava_to_unava_list)
                    for i in range(len(group_ava_to_unava)):
                        if len(group_ava_to_unava[i]) == 0:
                            zero_row = np.zeros((1, traffic_matrix_permutated.shape[1]), dtype=traffic_matrix_permutated.dtype)
                            traffic_matrix_permutated_row_sorted = np.vstack((traffic_matrix_permutated[:i, :], zero_row, traffic_matrix_permutated[i:, :]))
                            zero_col = np.zeros((traffic_matrix_permutated_row_sorted.shape[0], 1), dtype=traffic_matrix_permutated_row_sorted.dtype)
                            traffic_matrix_permutated_row_col_sorted = np.hstack((traffic_matrix_permutated_row_sorted[:, :i], zero_col, traffic_matrix_permutated_row_sorted[:, i:]))
                            traffic_matrix_permutated = traffic_matrix_permutated_row_col_sorted
                    
                    current_traffic_matrix = current_traffic_matrix + traffic_matrix_permutated
                    current_traffic_matrix_64_64 = current_traffic_matrix_64_64 + traffic_matrix_64_64

                    #t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava, first, traffic_matrix_permutated,))  # required processing time
                    #t.start()

                    print(job_queue[0])
                    
                    updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), traffic_matrix_permutated, ava_to_unava, traffic_matrix_64_64)
                    all_jobs[first_num] = (first_num, updated_job_done)

                    t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated, traffic_matrix_64_64,))  # required processing time
                    t.start()

                    print(job_queue[0])
                    job_queue.pop(0)
                else:
                    #t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava, first, -1,))  # required processing time
                    #t.start()
                    updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.now(), -1, ava_to_unava, -1)
                    print("HERE!!")
                    all_jobs[first_num] = (first_num, updated_job_done)

                    t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, -1,-1,))
                    # t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava.copy(), first, traffic_matrix_permutated,))
                    t.start()

                    print(job_queue[0])
                    job_queue.pop(0)
            else:
                continue
            
            
            
            # OCSアルゴリズム最大強化(自由FullFat)終了

            # Hybrid

            # Hybrid


            # 制約強化版
            """
            
            group_idle_counts = []
            # group_large_ava_counts = []
            if (node_count >= first_cpu and first_cpu <= 16):
                for group in range(4):
                    idle_count = 0
                    for node_num in range(group * 16,(group + 1) * 16):
                        if RG.nodes[node_num]["ava"] == "yes":
                            idle_count += 1
                    group_idle_counts.append(idle_count)
                

                filtered_counts = [count_ for count_ in group_idle_counts if count_ >= first_cpu ]

                if filtered_counts:
                    min_idle = min(filtered_counts)
                    candidate_groups = [i_ for i_ , cnt in enumerate(group_idle_counts) if cnt == min_idle and cnt >= first_cpu]
                    selected_group = candidate_groups[0]
                    idle_nodes = [i for i in range(selected_group*8, (selected_group+1)*8) if RG.nodes[i]["ava"] == "yes"]
                    chosen_idle_nodes = [idle_nodes[i] for i in range(first_cpu)]
                    for chosen_node in chosen_idle_nodes:
                        RG.nodes[chosen_node]["ava"] = "no"
                        nodelist_to_draw.append(chosen_node)
                        ava_to_unava.append(chosen_node)
                    t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava, first,))  # required processing time
                    t.start()
                    updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.datetime.now(), -1)
                    all_jobs[first_num] = (first_num, updated_job_done)
                    job_queue.pop(0)
                    continue
                else:
            """
            """
                    for group in range(4):
                        idle_count = 0
                        for node_num in range(group * 16, (group + 1) * 16):
                            if RG.nodes[node_num]["ava"] == "yes":
                                idle_count += 1
                        group_idle_counts.append(idle_count)
            """
            """
                    sorted_group_idle_counts = sorted(group_idle_counts, reverse = True)
                    for i in renge(4):
                        if ((i + 1) * sorted_group_idle_counts[i] >= first_cpu):
                            one_rack_cpu = first_cpu // (i + 1)
                            one_rack_cpu_arr = [one_rack_cpu for n in range(i + 1)]
                            one_rack_cpu_ex = first_cpu % (i + 1)
                            for j in range(one_rack_cpu_ex):
                                one_rack_cpu_arr[j] = one_rack_cpu_arr[j] + 1
                            one_rack_cpu_arr_reversed = sorted(one_rack_cpu_arr, reverse = True)
                            for n in one_rack_cpu_arr_reversed:
                                filtered_counts = [count_ for count_ in group_idle_counts if count_ >= n]
                                if filtered_counts:
                                    min_idle = min(filtered_counts)
                                    candidate_groups = [i_ for i_ , cnt in enumerate(group_idle_counts) if cnt == min_idle and cnt >= n]
                                    selected_group = candidate_groups[0]
                                    idle_nodes = [i_ for i_ in range(selected_group*16, (selected_group+1)*16) if RG.nodes[i_]["ava"] == "yes"]
                                    chosen_idle_nodes = [idle_nodes[i] for i in range(int(n))]
                                    for chosen_node in chosen_idle_nodes:
                                        RG.nodes[chosen_node]["ava"] = "no"
                                        nodelist_to_draw.append(chosen_node)
                                        ava_to_unava.append(chosen_node)
                                    group_idle_counts[selected_group] = group_idle_counts[selected_group] - n
                                else:
                                    continue:
                        else:
                            continue:
                    continue
            elif (node_count >= first_cpu and first_cpu > 16):
                print(first_cpu)
                for group in range(8):
                    idle_count = 0
                    for node_num in range(group * 8,(group + 1) * 8):
                        if RG.nodes[node_num]["ava"] == "yes":
                            idle_count += 1
                    group_idle_counts.append(idle_count)
                sorted_group_idle_counts = sorted(group_idle_counts, reverse=True)
                for i in range(8):
                    if ((i + 1) * sorted_group_idle_counts[i] >= first_cpu):
                        one_rack_cpu = first_cpu // (i + 1)
                        one_rack_cpu_arr = [one_rack_cpu for n in range(i + 1)]
                        one_rack_cpu_ex = first_cpu % (i + 1)
                        for j in range(one_rack_cpu_ex):
                            one_rack_cpu_arr[j] = one_rack_cpu_arr[j] + 1
                        print(one_rack_cpu_arr)
                        for n in one_rack_cpu_arr:
                            filtered_counts = [count_ for count_ in group_idle_counts if count_ >= n ]
                            if filtered_counts:
                                min_idle = min(filtered_counts)
                                candidate_groups = [i_ for i_ , cnt in enumerate(group_idle_counts) if cnt == min_idle and cnt >= n]
                                selected_group = candidate_groups[0]
                                idle_nodes = [i_ for i_ in range(selected_group*8, (selected_group+1)*8) if RG.nodes[i_]["ava"] == "yes"]
                                chosen_idle_nodes = [idle_nodes[i] for i in range(int(n))]
                                print(chosen_idle_nodes)
                                for chosen_node in chosen_idle_nodes:
                                    RG.nodes[chosen_node]["ava"] = "no"
                                    nodelist_to_draw.append(chosen_node)
                                    ava_to_unava.append(chosen_node)
                                group_idle_counts[selected_group] = group_idle_counts[selected_group] - n
                            else:
                                continue
                        t = threading.Timer(first_time, unlock_unavailable,(ava_to_unava, first,))  # required processing time
                        t.start()
                        print(job_queue[0]) 
                        updated_job_done = (job_queue[0][1][0], job_queue[0][1][1], job_queue[0][1][2], job_queue[0][1][3], datetime.datetime.now(), -1)
                        all_jobs[first_num] = (first_num, updated_job_done)
                        job_queue.pop(0)
                        break
                    else:
                        continue
            else:
                continue
            
            """

            # 制約強化版終了


            """
            if (transform == False and to_first == True and fso_not_found == False):
                if (GUI.schedule == "LIFO"):
                    job_queue.insert(0, job_queue.pop(-1))
                first = job_queue[0]
                first_num = job_queue[0][0]
                first_cpu = job_queue[0][1][0]
                first_time = job_queue[0][1][1]

                if (first_cpu < 1 or first_cpu > size_grid_x * size_grid_y or first_time < 0):
                    print(datetime.datetime.now(), "job: ", first, " can not be scheduled due to errorous requests")
                    job_queue.pop(0)
                    reset()
                    #         checkover()
                    to_first = True
                    continue

                    # 150819 huyao pure fso
            if (GUI.mode == "FSO"):
                fso_not_found = fso(first, first_cpu, first_time)
                continue

            if (GUI.mode == "QUBO"):
                fso_not_found = qubo_allocation(first, first_cpu, first_time)
                continue

            if (transform == True):
                #             while(True):
                #                 if(lock==False):
                #                     break
                g = divi(first_cpu + fill, temp_y + 1)
                x = g[1]
                y = g[0]
                if (x == 1 and y > 2):
                    #             transform = False
                    reset()
                    #                 if(GUI.mode=="FSO"):
                    #                     fso()
                    to_first = False
                    continue
            else:
                g = divi(first_cpu + fill)  # required cpus
                x = g[1]  # length
                y = g[0]  # width

            while (x > size_grid_x):
                g = divi(first_cpu + fill, y + 1)
                x = g[1]
                y = g[0]

            if (y > size_grid_y):
                if (transform == False):
                    print(datetime.datetime.now(), "job: ", first, " can not be scheduled due to lack of resources")
                    job_queue.pop(0)
                    to_first = True
                #             checkover()
                else:
                    to_first = False
                reset()
                continue

            if (x == 1 and y > 2):
                fill = 1
                g = divi(first_cpu + fill)
                x = g[1]  # length
                y = g[0]  # width
            found = False  # allocated cpus
            for yy in range(size_grid_y - y + 1):  # left top vertex of x*y grid
                for xx in range(size_grid_x - x + 1):
                    flag = True  # if each cpu is available in x*y grid
                    flag_ = True  # useful if fill!=0
                    for xxx in range(xx, xx + x):  # ergodic in x*y grid
                        for yyy in range(yy, yy + y):
                            if (RG.nodes[(xxx, yyy)]["ava"] == "no" and fill == 0):
                                flag = False
                                break
                            if (RG.nodes[(xxx, yyy)]["ava"] == "no" and fill == 1):
                                if (flag_ == True):
                                    flag_ = False
                                else:
                                    flag = False
                                    break

                        if (flag == False):
                            break
                    if (flag == True):
                        print(datetime.datetime.now(), "job: ", first, " is scheduled to the nodes:")
                        all = True
                        ava_to_unava = []
                        for xxx in range(xx, xx + x):
                            for yyy in range(yy, yy + y):
                                if (xxx == xx + x - 1 and yyy == yy + y - 1 and fill == 1 and all == True and
                                        RG.nodes[(xxx, yyy)]["ava"] == "yes"):
                                    #                                 print xxx, yyy
                                    break
                                if (RG.nodes[(xxx, yyy)]["ava"] == "yes"):
                                    RG.nodes[(xxx, yyy)]["ava"] = "no"
                                    #                             print "(", xxx, ", ", yyy, ") "
                                    nodelist_to_draw.append((xxx, yyy))
                                    # print RG.node[(xxx,yyy)]
                                    ava_to_unava.append((xxx, yyy))
                                #                             t = threading.Timer(jobs_[0][1][1], unlock, (RG.node[(xxx,yyy)], xxx, yyy, xx, yy, jobs_[0],)) #required processing time
                                #                             t.start()
                                else:
                                    all = False
                        t = threading.Timer(first_time, unlock_unavailable,
                                            (ava_to_unava, first,))  # required processing time
                        t.start()
                        job_queue.pop(0)
                        found = True
                        #                 transform = False
                        #                 fill = 0
                        reset()
                        to_first = True
                        break
                    if (yy == size_grid_y - y and xx == size_grid_x - x):
                        # 150821 huyao check if any other job inserted during transform
                        #                     if(first_num != queue[0][0]):
                        #                         reset()
                        #                         break
                        # print datetime.datetime.now(), "job: ", jobs_[0], " can not be scheduled due to no available resources"
                        # jobs_.pop(0)    #temp
                        # 150828 huyao
                        if (first_cpu == 2):
                            reset()
                            to_first = False
                        transform = True
                        #                     lock = True
                        temp_x = x
                        temp_y = y
                if (found == True):
                    break
                    # print len(jobs_)
                    """
        elif (all_jobs_submitted == True):
            break

def draw_image():
    pos = dict(list(zip(RG, RG)))
    nx.draw(RG, pos, node_size=30, with_labels=True)
    # nx.draw_networkx_nodes(RG,pos,nodelist=[(0,0)],node_color='b')
    nx.draw_networkx_nodes(RG, pos, nodelist=nodelist_to_draw, node_color='b')
    plt.setp(plt.gca(), 'ylim', list(reversed(plt.getp(plt.gca(), 'ylim'))))
    # plt.setp(plt.gca(), 'xlim', list(reversed(plt.getp(plt.gca(), 'xlim'))))
    # plt.show(block = False)
    plt.show()

def simulation_main():
    initialize_graph()
    initialize_all_jobs_list()
    initialize_job_queue()

    # start two threads
    start_jobs_submitting_thread()
    start_dostat_thread()

    # job allocation
    loop_allocate_all_jobs()

    if (False):
        draw_image()

    result_ave_wait = 0
    global job_records, start_time, communication_volume, communication_volume_64_64
    job_records = []
    communication_volume_array = []
    communication_volume_array_64_64 = []


    for i in range(0, NUM_SIMULATION_JOBS - 1):
        start_time = all_jobs[i][1][4]  # datetime 型
        submit_time = all_jobs[i][1][3]  # datetime 型
        # print(all_jobs[i])
        job_records.append(all_jobs[i])
        #print(start_time)
        #print(submit_time)
        wait_time_sub = (start_time - submit_time).total_seconds()
        result_ave_wait += wait_time_sub
        if isinstance(all_jobs[i][1][5], np.ndarray):
            communication_volume_array.append(all_jobs[i][1][5].sum() / 4)
        if isinstance(all_jobs[i][1][7], np.ndarray):
            communication_volume_array_64_64.append(all_jobs[i][1][7].sum() / 64)
    communication_volume_64_64 = max(communication_volume_array_64_64)
    communication_volume = max(communication_volume_array)

    global result_utility, end_time, fps, min_time, im, im_64_64, time_traffic_list
    result_ave_wait /= NUM_SIMULATION_JOBS
    result_utility = result_utility * 100 / 64 / result_wait_sum
    print("", flush=True)
    print("-------------", size_grid_x, "wait_sum: ", result_wait_sum.real / 100.0)
    print(result_wait_sum)
    print("average_wait_time", result_ave_wait)
    print("utillity", result_utility)

    timeline_step_seconds = 00.1
    fps = 1000 / timeline_step_seconds

    start_time = min(job[1][4] for job in job_records)
    end_time = max(job[1][4] + timedelta(seconds=job[1][1]) for job in job_records)

    total_duration_seconds = (end_time - start_time).total_seconds()
    num_steps = int(total_duration_seconds / timeline_step_seconds) + 1

    # timeline = list(start_time + timedelta(seconds=i) for i in range(int((end_time - start_time).total_seconds()) + 1))

    timeline = []
    time_traffic_list = []
    current_time = start_time
    for _ in range(num_steps): # i は使わないので _ で
        timeline.append(current_time)
        current_time += timedelta(seconds=timeline_step_seconds)
    if timeline and timeline[-1] < end_time:
        timeline.append(end_time)

    all_unique_nodes = set()
    for _, job_info in job_records:
        for node in job_info[6]:
            all_unique_nodes.add(node)

    heatmap_node_labels = [f'group{i+1}' for i in range(4)]
    ax_heatmap.set_xticks(np.arange(4))
    ax_heatmap.set_yticks(np.arange(4))
    ax_heatmap.set_xticklabels(heatmap_node_labels)
    ax_heatmap.set_yticklabels(heatmap_node_labels)
    ax_heatmap.set_title("Communication Matrix")
    ax_heatmap.set_xlabel("To Node")
    ax_heatmap.set_ylabel("From Node")

    heatmap_node_64_64_labels = [f'node{i+1}' for i in range(64)]
    ax_heatmap_64_64.set_xticks(np.arange(64))
    ax_heatmap_64_64.set_yticks(np.arange(64))
    ax_heatmap_64_64.set_xticklabels(heatmap_node_64_64_labels)
    ax_heatmap_64_64.set_yticklabels(heatmap_node_64_64_labels)
    ax_heatmap_64_64.set_title("Communication Matrix 64*64")
    ax_heatmap_64_64.set_xlabel("To Node")
    ax_heatmap_64_64.set_ylabel("From Node")



    initial_heatmap_data = np.zeros((4, 4))
    initial_heatmap_data_64_64 = np.zeros((64, 64))
    im = ax_heatmap.imshow(initial_heatmap_data, cmap='viridis', origin='upper', vmin=0, vmax=communication_volume) # vmaxは通信量の最大値に応じて調整
    im_64_64 = ax_heatmap_64_64.imshow(initial_heatmap_data_64_64, cmap='viridis', origin='upper', vmin=0, vmax=communication_volume_64_64)
    cbar = fig.colorbar(im, ax=ax_heatmap, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_64_64 = fig.colorbar(im_64_64, ax=ax_heatmap_64_64, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Communication Volume')
    cbar_64_64.set_label('Communication Volume 64*64')

    # current_time_text = ax_graph.text(0.01, 0.99, '', transform=ax_graph.transAxes, va='top', ha='left', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    all_start_times = [job_info[4] for _, job_info in job_records]
    all_end_times = [job_info[4] + timedelta(seconds=job_info[1]) for _, job_info in job_records]
    min_time = min(all_start_times) - timedelta(seconds=5) # 開始より少し前
    max_time = max(all_end_times) + timedelta(seconds=5) # 終了より少し後

    total_animation_seconds = (max_time - min_time).total_seconds()
    animation_interval_ms = 100 # 各フレーム間のミリ秒
    fps = 1000 / animation_interval_ms
    num_frames = int(total_animation_seconds * fps)

    # fig, ax = plt.subplots(figsize=(6, 6))
    anim = FuncAnimation(fig, update_2, frames=num_frames, interval=animation_interval_ms, repeat=False, blit=False)
    # plt.show()

    # ガントチャート部分

    all_nodes = set()
    for job_id, job_info in job_records:
        nodes_for_job = job_info[6] # ノードのリストを取得
        for node in nodes_for_job:
            all_nodes.add(node)
    sorted_nodes = sorted(list(all_nodes))
    node_y_pos = {node: i for i, node in enumerate(sorted_nodes)}

    tasks = sorted(list(set(job_id for job_id, job_info in job_records)))
    colors = plt.cm.get_cmap('tab10', len(tasks))
    task_colors = {task: colors(i) for i, task in enumerate(tasks)}

    for job_id, job_info in job_records:
        start_dt = job_info[4]
        runtime_in_seconds = job_info[1]
        task_name = job_id
        task_color = task_colors[task_name]

        duration_in_days = runtime_in_seconds / (24 * 3600)

        nodes_for_job = job_info[6]

        for node in nodes_for_job:
            y_position = node_y_pos[node] # そのノードのY軸位置を取得

            # barh で棒を描画
            # label= を削除 (凡例は出さないため)
            ax_gantt.barh(y_position, duration_in_days, left=start_dt, height=0.6,
                    align='center', color=task_color)
            
            # ジョブ名（タスク名）を表示
            # 棒の中央にテキストを配置
            text_x_pos = start_dt + timedelta(seconds=runtime_in_seconds / 2)
            ax_gantt.text(text_x_pos, y_position, task_name,
                    va='center', ha='center', fontsize=9, color='white',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
    ax_gantt.set_yticks(list(node_y_pos.values()))
    ax_gantt.set_yticklabels(list(node_y_pos.keys()))
    ax_gantt.set_xlabel("Time")
    ax_gantt.set_ylabel("Node")
    ax_gantt.set_title("Job Schedule on Nodes Gantt Chart by Task Type")
    ax_gantt.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
    ax_gantt.xaxis.set_major_locator(mdates.AutoDateLocator())

    fig_gantt.autofmt_xdate()

    plt.tight_layout() # rect引数を削除して自動調整に任せる
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    fig_gantt.savefig('gantt_chart.png', dpi=300, bbox_inches='tight') # 例: fig変数にFigureが格納されているとして
    plt.close(fig_gantt) # オプション: 保存後、このFigureを閉じてメモリを解放

    anim.save('animation.gif', writer='pillow', fps=20, dpi=100)
    # plt.show()

    print("Node-based Gantt Chart script finished.")
    print(len(time_traffic_list))

    output_csv_file = 'time_traffic_matrix.csv'
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # ヘッダー行（任意）
        header = ['Time', 'Matrix']
        writer.writerow(header)

        for time, comm_matrix, comm_matrix_64_64 in time_traffic_list:
            row = [time]
            # 4x4行列を1次元にフラットにして追加
            row.extend(comm_matrix.flatten().tolist())
            row.extend(comm_matrix_64_64.flatten().tolist())
            writer.writerow(row)

    # ガントチャート部分終了

    # nx.draw(RG, with_labels=True)
    # plt.show()

"""
def update(frame):
    global job_records, pos
    ax.clear()
    active_nodes = {}
    for job_id, (cores, runtime, _, _, start, _, nodes) in job_records:
        end = start + timedelta(seconds=runtime)
        if start <= frame < end:
            for n in nodes:
                active_nodes[n] = job_id

    node_colors = []
    labels = {}
    for n in RG.nodes:
        if n in active_nodes:
            node_colors.append("red")
            labels[n] = str(active_nodes[n])
        else:
            node_colors.append("lightgray")
            labels[n] = ""

    nx.draw_networkx(RG, pos=pos, ax=ax, with_labels=True, labels=labels,
                     node_color=node_colors, node_size=500, font_size=10, font_color='black')
    ax.set_title(f"Time: {frame.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=14, color='black')
    # ax.set_title(f"Time: {frame.strftime('%H:%M:%S')}")
    ax.axis("off")
"""
"""
def update(frame):
    global job_records, pos
    ax_graph.clear()
    ax_heatmap.clear()
    active_nodes = {}
    for job_id, (cores, runtime, _, _, start, _, nodes) in job_records:
        end = start + timedelta(seconds=runtime)
        if start <= frame < end:
            for n in nodes:
                active_nodes[n] = job_id

    node_colors = []
    labels = {}
    for n in RG.nodes:
        if n in active_nodes:
            node_colors.append("red")
            labels[n] = str(active_nodes[n])
        else:
            node_colors.append("lightgray")
            labels[n] = ""

    nx.draw_networkx(RG, pos=pos, ax=ax, with_labels=True, labels=labels,
                     node_color=node_colors, node_size=500, font_size=10, font_color='black')
    ax_graph.set_title(f"Time: {frame.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=14, color='black')
    ax_graph.set_axis_off()

    # ax.set_title(f"Time: {frame.strftime('%H:%M:%S')}")
    ax.axis("off")
"""
def animate(frame):
    global start_time, end_time, timeline_step_seconds, fps
    current_datetime = start_time + timedelta(seconds=frame / fps)
    current_time_text.set_text(f'Time: {current_datetime.strftime("%H:%M:%S")}')

    # --- NetworkXグラフのノード色の更新 ---
    node_colors = []
    # node_colors = [default_node_color] * len(graph_nodes) # 全てのノードをデフォルト色で初期化
    current_heatmap_data = np.zeros((4, 4)) # ヒートマップデータを初期化

    # active_job_info_list = [] # 現在時刻に実行中のジョブのリスト

    for job_id, job_info in job_records:
        start_dt = job_info[4]
        runtime_in_seconds = job_info[1]
        task_name = job_id
        # task_color = task_colors[task_name]
        communication_matrix = job_info[5] # 4x4通信行列
        nodes_for_job = job_info[6]

        end_dt = start_dt + timedelta(seconds=runtime_in_seconds)

        if start_dt <= current_datetime < end_dt:
            # ジョブが現在アクティブな場合
            active_job_info_list.append(job_info)

            # グラフのノード色を更新
            for node in nodes_for_job:
                if node in graph_nodes: # NetworkXグラフに存在するノードのみ
                    node_idx = graph_nodes.index(node)
                    node_colors[node_idx] = task_color # アクティブなノードをタスクの色に
            
            # ヒートマップのデータに通信行列を加算
            # ここで、通信行列が4x4であるため、node1-node4に対応していると仮定
            # もしgraph_nodesがnode5などを含んでいても、heatmap_node_labelsの範囲で処理
            current_heatmap_data += communication_matrix
    time_traffic_set = [current_datetime, current_heatmap_data]
    time_traffic_list.append(time_traffic_set)

    # NetworkXノードの色を実際に更新
    node_collection.set_facecolor(node_colors)

    # --- ヒートマップの更新 ---
    im.set_array(current_heatmap_data) # 合計した行列でヒートマップを更新

    # blit=True の場合は変更されたアーティストを返す必要がある
    # 今回は node_collection, im, current_time_text が変更される
    return [node_collection, im, current_time_text]

def update_2(frame_idx):
    # 現在時刻を計算
    current_datetime = min_time + timedelta(seconds=frame_idx / fps)

    # --- NetworkXグラフの更新 (ax_graphに対する処理) ---
    ax_graph.clear() # グラフをクリア
    ax_graph.set_title(f"Time: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=14, color='black')
    ax_graph.axis("off") # 軸は非表示にする

    active_nodes_map = {} # {node_name: job_id}
    current_heatmap_data = np.zeros((4, 4)) # ヒートマップデータもクリアして初期化
    current_heatmap_data_64_64 = np.zeros((64, 64))

    for job_id, job_info in job_records:
        start_dt = job_info[4]
        runtime = job_info[1] # seconds
        communication_matrix = job_info[5] # 4x4 matrix
        communication_matrix_64_64 = job_info[7]
        nodes_for_job = job_info[6]

        end_dt = start_dt + timedelta(seconds=runtime)

        if start_dt <= current_datetime < end_dt:
            # ジョブが現在アクティブな場合
            for n in nodes_for_job:
                # NetworkXグラフのノードカラー更新用
                active_nodes_map[n] = job_id
            
            # ヒートマップのデータに通信行列を加算
            current_heatmap_data += communication_matrix
            current_heatmap_data_64_64 += communication_matrix_64_64

    node_colors = []
    labels = {} # ラベルはアクティブなノードにのみjob_idを表示
    for n in RG.nodes:
        if n in active_nodes_map:
            node_colors.append("red") # アクティブなノードは赤
            labels[n] = str(active_nodes_map[n]) # アクティブなノードにjob_idを表示
        else:
            node_colors.append("lightgray") # 非アクティブなノードは灰色
            labels[n] = "" # 非アクティブなノードはラベルなし

    time_traffic_set = [current_datetime, current_heatmap_data, current_heatmap_data_64_64]
    time_traffic_list.append(time_traffic_set)
    # RG, pos はグローバルスコープで定義されているため、そのまま参照できます
    nx.draw_networkx(RG, pos=pos, ax=ax_graph, with_labels=True, labels=labels, # axをax_graphに変更
                     node_color=node_colors, node_size=2000, font_size=10, font_color='black',
                     edge_color='gray', alpha=0.5) # エッジも描画するならここに追加

    # --- ヒートマップの更新 (ax_heatmapに対する処理) ---
    im.set_array(current_heatmap_data) # ヒートマップのデータを更新
    im_64_64.set_array(current_heatmap_data_64_64)

    # blit=False の場合、戻り値は必須ではないですが、慣例として空リストを返すこともあります。
    return []

"""
def update(frame):
    ax.clear()
    # 現在の時刻をテキストで表示するだけの非常にシンプルなテスト
    ax.text(0.5, 0.5, f"Time: {frame.strftime('%H:%M:%S')}", 
            ha='center', va='center', fontsize=16, color='black')
    ax.set_title(f"Animation Test: {frame.strftime('%H:%M:%S')}", color='black')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
"""

simulation_main()
