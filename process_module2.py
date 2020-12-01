"""

map: 对数据进行分配
reduce: 对map计算的结果进行汇总

第2版 输入的参数数据处理加强

下一版：map和reduce可以共用Queue队列，共用Process进程，队列的传输数据指明是reduce还是map，然后process里面可以根据从Queue读到的
数据切换调用 map func 和 reduce func。

再考虑把通信消息用专门的队列，不和输入队列混淆到一起，
再考虑每个进程里面开几个线程，一个线程用于处理数据，一个线程用于与其他进程通信。

后续：要不每一个进程一个Queue管道，然后map的时候直接把计算数据送到对应的reduce进程
（统计reduce的key个数，然后每个key%管道数量就可以得到数据放到第几个管道，
然后同一个key的所有数据都由一个管道处理，同一个管道处理多个key，然后管道自己进行reduce，
再对每个key分别调用reduce_fn，放到结果管道中）。
问题，未知每个key的reduce时间，平均分任务可能不能充分利用资源。

"""
from multiprocessing import Process, Queue, Pool
import inspect
import functools
import itertools
from collections import abc, defaultdict
import math
import os
from copy import deepcopy
import logging
import reprlib
import time

logger = logging.getLogger(__name__)

def show_args(args):
    return "\n".join([name+" = "+str(value) for name, value in args.arguments.items()])

class Distributed():
    def __init__(self, num_workers=5, max_size=100):
        self.num_workers = num_workers
        self.actual_workers = num_workers
        self.max_size = max_size
        """
        目前实现是一个子进程写入input_queue, 多个子进程读取input_queue并处理，然后多进程结果写入output_queue，
        主进程从output_queue读数据并处理，如果主进程处理很慢，比如写入文件，那么会造成即便是流式处理，也会造成
        output_queue内存爆炸，因为多个进程写入output_queue比读取快，output_queue数量始终增加。
        修改：加上Queue的大小限制，防止写入速度过快，导致内存崩溃。
        [TODO] 目前的输入端实际上没有支持流式处理，因为要获取整个数据的大小，可以改进成支持传入迭代器，实现真正的流式并行
        """
        self.input_queue = Queue()
        self.output_queue = Queue(maxsize=max_size) 
        self.message_queue = Queue()
        self.reduce_input_queue = Queue()
        # self.reduce_output_queue = Queue(max_size=max_size)
        self.group_by = None
        self.reduce_fn = None

    def prepare_input(self, dist_kwargs, chunk_size=1):
        length = len(next(iter(dist_kwargs.values())))
        for i in range(0, length, chunk_size):
            dist_tmp = {k:v[i:i+chunk_size] for k,v in dist_kwargs.items()}
            yield dist_tmp
        
    def prepare_iterable_input(self, dist_kwargs, chunk_size=1):
        """
        [NOTE] 
        每次yield值都共享这个kwargs，如果每次都修改kwargs，则会导致已经放入Queue的数据被修改
        每次yield不能通过inplace这个kwargs来改变值，但是如果non distributed argument都要copy一下，
        至少每次yield值kwargs不是指向同一个kwargs, 则太耗内存了，当前不考虑内存消耗
        优化措施应该是，non distributed argument正常共享，而distributed argument 单独创建变量，不借助non distributed argument。
        即返回两个值，一个是每次yield都共用的引用，一个是每次yield都创建的引用。

        后来发现干脆不处理non distributed kwargs，因为传入进来又传出去，没变什么，交给write_process去干
        """
        dist_keys = list(dist_kwargs.keys())
        tmp = []
        count = 0
        for values in zip(*dist_kwargs.values()):
            tmp.append(values)
            count += 1
            if count >= chunk_size:
                count = 0 
                # [(10,20,), (2,1)] => [(10,2), (20,1)]
                """[TODO] 类型应该和原始数据类型保持一致，这里是v[1:]元组？是元组也没关系，反正是遍历"""
                dist_tmp = {v[0]:v[1:] for v in zip(dist_keys, *tmp)}
                tmp =[]
                # kwargs.update(dist_tmp) # 如果共享kwargs, 那么修改kwargs会修改Queue中已经放入的数据
                # 所以干脆分开传入
                yield dist_tmp
        if count > 0:
            dist_tmp = {v[0]:v[1:] for v in zip(dist_keys, *tmp)}
            tmp =[]
            yield dist_tmp

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and kwargs == {} and isinstance(args[0],abc.Callable):
            return self.multiprocess()(args[0])    
        return self.multiprocess(*args, **kwargs)

    def multiprocess(self, map_keys=-1, chunk_size=1, sized=False, time_out=20, group_by=0, reduce_fn=None, stream=True, use_sid=False, stream_handler=None,
        auto_strict=True,
        num_workers=5,
        max_size=100,
    ):
        """
        [NOTE]
        1. 当数据是list，总长度已知，是否自动选择最优的chunk_size:=length/num_process？No ~
        2. time_out缺省值是30是有理由的，如果主进程出现意外并且time_out是None(表示永不超时)，即不发送0作为子进程关闭的条件，那么子进程就可能一直挂着，所以推荐设置一个延时自动关掉
            所以如果子进程启动后，程序数据处理要很久才送到管道，导致后面的子进程获取不到数据超时退出，那么用户自己主动设置超时时间长一点，因为主进程正常运行，会及时发送0通知子进程及时结束，
            无需等待time_out才退出。
        3. auto_strict 意味着当stream=False并且需要进行reduce的时候，这个时候不是标准map_reduce，不同的chunk_size很可能会导致不同map结果，从而影响reduce过程
            甚至会出错，除非自己知道正在做什么，否则不建议这么做。
        """
        self.actual_workers = num_workers
        self.num_workers = num_workers
        if self.max_size != max_size:
            self.max_size = max_size
            self.output_queue = Queue(maxsize=max_size)
        if isinstance(map_keys, str):
            map_keys = (map_keys,)
        def decorator_fn(func):
            sig = inspect.signature(func)
            @functools.wraps(func)
            def wrap_func(*args, **kwargs):
                nonlocal map_keys, chunk_size, sized, group_by, reduce_fn, stream, use_sid, time_out, stream_handler
                reduce_fn = reduce_fn or self.reduce_fn 
                #stream = stream
                # print(stream)
                will_reduce = reduce_fn is not None and stream_handler is None
                def map_func():
                    pid = os.getpid()
                    logger.info("map process %d start...", pid)
                    while True:
                        try:
                            # time_out时间内进程没有读取到数据也直接退出
                            data = self.input_queue.get(timeout=time_out)
                        except:
                            logger.error("process {} timeout exit...".format(pid))
                            return 0
                        if data == 0: # 读取到0表示map要终止了,reduce尝试开始
                            logger.info("process {} get 0 exit map...".format(pid))
                            break        
                        sid, non_dist, dist = data
                        result = func(**non_dist, **dist)
                        self.output_queue.put((sid, result))
                    if will_reduce: # 满足这个条件，reduce才会启动
                        logger.info("process {} try to start reduce...".format(pid))
                        while True:
                            try:
                                # [NOTE] 主进程汇聚map的结果再进行分配，会不会让reduce的进程等待时间过长？尝试增加time_out
                                # 如果共用同一个input_queue，那么会不会存在 0 被一个进程读了两次？导致异常，所以用两个队列吧
                                data = self.reduce_input_queue.get(timeout=time_out)
                            except:
                                logger.error("process {} timeout exit...".format(pid))
                                return 0
                            if data == 0:
                                logger.info("process {} get 0 exit reduce ...".format(pid))
                                break
                            result = reduce_fn(*data)
                            # 共享output_queue？
                            self.output_queue.put((result))

                    # return result
                default_kwargs = {}
                for name, param in sig.parameters.items():
                    if param.default != inspect._empty: # 可选位置参数,如果用户调用不传入这个，signature捕捉不到的
                        default_kwargs[name] = param.default
                bound_args = sig.bind(*args, **kwargs)
                default_kwargs.update(bound_args.arguments)
                logger.debug("input args:"+show_args(bound_args))
                logger.debug("all args:"+reprlib.repr(default_kwargs))
                non_dist_argument = {}
                dist_argument = {}
                if map_keys == -1:
                    map_keys = default_kwargs.keys()
                logger.debug("distributed keys %s", map_keys)
                sized_input = False
                for name,value in default_kwargs.items():
                    if name in map_keys and isinstance(value, abc.Iterable):
                        if isinstance(value, abc.Sized):
                            sized_input = True
                        dist_argument[name] = value
                    else:
                        non_dist_argument[name] = value
                if sized_input and sized:
                    # 注意即使把chunk_size设置成 1 , 实际输入函数的数据也是一个元组，只不过只有一个样本而已，而不是一个简单的样本
                    data_generator = self.prepare_input(dist_kwargs=dist_argument, chunk_size=chunk_size)
                else:
                    data_generator = self.prepare_iterable_input(dist_kwargs=dist_argument, chunk_size=chunk_size)
                
                def write_func():
                    count = 0
                    for i, data_chunk in enumerate(data_generator):
                        logger.debug("input %s", str(data_chunk)[:40])
                        self.input_queue.put((i*chunk_size, non_dist_argument, data_chunk))
                        count += 1
                    # 此时所有的数据都进到input_queue了吗？如果主进程再从message_queue中获取消息后
                    # 往input_queue传入数据会不会出现插队现象呢？
                    # 似乎会插队，改为由本进程写入终止符 0 ，这样确保写入map数据全部写入之后才会写入 0
                    self.message_queue.put(count)
                    for _ in range(self.actual_workers):
                        self.input_queue.put(0)
                    logger.info("write process stop, total %d...", count)
                
                process = []
                for _ in range(self.actual_workers):
                    p = Process(target=map_func, )
                    process.append(p)
                write_process = Process(target=write_func,)
                logger.info("write process start...")
                write_process.start()
                for i, p in enumerate(process):
                    p.start()

                count = None
                while count is None:
                    count = self.message_queue.get()
                logger.info("num of map tasks ( also chunked data nums, or batch nums ): %d", count)
                
                items = []
                # 注意这个items是不设置大小的，否则可能会等不到数据卡死
                # 但是不注意这个，万一前面的某个数据一直没有等到，内存就会崩溃，因为一直在缓存数据...
                # [TODO] 可以改为 priority queue?
                if stream == False:
                    logger.warning("Non standard mapreduce! the map results will be different if using different chunk-size")
                
                if stream or (will_reduce and auto_strict):
                    def sorted_stream_gen():
                        index = 0
                        yield_index = 0
                        while yield_index < count:
                            for item in items:
                                if item[0] == yield_index*chunk_size:
                                    yield from item[1]
                                    items.remove(item)
                                    yield_index += 1
                                    break
                            if index < count:
                                item = self.output_queue.get()
                                index += 1
                                if item[0] == yield_index*chunk_size:
                                    yield from item[1]
                                    yield_index += 1
                                else:
                                    items.append(item)
                else:
                    def sorted_stream_gen():
                        index = 0
                        yield_index = 0
                        while yield_index < count:
                            for item in items:
                                if item[0] == yield_index*chunk_size:
                                    yield item[1]
                                    items.remove(item)
                                    yield_index += 1
                                    break
                            if index < count:
                                item = self.output_queue.get()
                                print(item)
                                index += 1
                                if item[0] == yield_index*chunk_size:
                                    yield item[1]
                                    yield_index += 1
                                else:
                                    items.append(item)
                if stream_handler is not None:
                    # 注意，input_queue有专门的写入进程，如果主进程也写数据，会造成干扰，比如提前结束
                    # 所以主进程最好不写数据，至少数据写入进程没有写完之前，不要写入终止符0
                    # 所以最好是由写入进程专门写入终止符0，但是好像reduce就无法复用进程了？
                    # [TODO] 暂时这么做吧
                    # for _ in range(self.actual_workers):
                    #    self.input_queue.put(0)      
                    return stream_handler(sorted_stream_gen())
                # 对结果的收集，开始的实现是创建空list，然后append，最后再根据SID的信息排序。
                # 数量较多时，排序也会是性能的瓶颈，如果更快的有序呢？考虑二分查找插入？
                # No! 因为每个输出都有自己的SID，SID都是从0开始一直到多少个MAP程序的，所以
                # 实现创建一个大数组，然后每读取一个就指定索引赋值。
                if reduce_fn is None:
                    if stream : 
                        # stream下，多少输入就会有多少输出，这个results可能会比实际的大，因为最后一个chunked data是不够大的，
                        # 取决于数据量是否大于chunked_size。
                        # 注意用户的输出不能是 "@_@"，不然目前无法判断哪个才是最后一个
                        results = ["^_^" for i in range(count*chunk_size)]
                        get_item = 0
                        while get_item < count:
                            item = self.output_queue.get()
                            for i, v in enumerate(item[1]):
                                results[item[0]+i] = v # 因为item[0]是每一个chunk的首元素的SID
                            get_item += 1
                        i = -1
                        while results[i] == "^_^":
                            i -= 1
                        if i != -1:
                            results = results[:i+1]
                        #print(results)
                    else:
                        # 非 stream 模式，结果数量取决 分割的 chunk 的数量（根据chunk_size和总数量决定），相信用户每一组chunk对应的输出不需要分拆
                        results = [None for i in range(count)]
                        get_item = 0
                        while get_item < count:
                            item = self.output_queue.get()
                            results[item[0]//chunk_size] = item[1]
                            get_item += 1
                    # map 进程结束已经在数据写入那里解决了
                    # for _ in range(self.actual_workers):
                    #     self.input_queue.put(0)
                    return results
                
                ##################
                # reduce 正式开始
                ##################
                if auto_strict and stream == False:
                    logger.warning("auto_strict is working...shift the stream to True.")
                    stream = True
                # 启动reduce程序
                if isinstance(group_by, int):
                    group_fn = lambda r: r[group_by]
                elif isinstance(group_by, abc.Iterable):
                    group_fn = lambda r: tuple(r[i] for i in group_by)
                elif isinstance(group_by, abc.Callable):
                    group_fn = group_by
                
                reduce_inputs = defaultdict(list)
                for i, map_output in enumerate(sorted_stream_gen()):
                    sid = i*chunk_size
                    logger.debug("map output: %s, %s", sid, str(map_output)[:50])
                    if use_sid:
                        reduce_inputs[group_fn(map_output)].append((sid, map_output))
                    else:
                        reduce_inputs[group_fn(map_output)].append(map_output)
                logger.debug("reduce input example: %s", reprlib.repr(next(iter(reduce_inputs.items()))))
                
                def reduce_write_func():
                    count = 0
                    for i, data_chunk in enumerate(reduce_inputs.items()):
                        #print("input", data_chunk)
                        self.reduce_input_queue.put(data_chunk)
                        count += 1
                    self.message_queue.put(count)
                write_process = Process(target=reduce_write_func,)
                write_process.start()
                count = None
                while count is None:
                    count = self.message_queue.get()
                logger.debug("num of reduce tasks: %d", count)
                results = []
                while len(results) < count:
                    results.append(self.output_queue.get())
                for _ in range(self.actual_workers):
                    self.reduce_input_queue.put(0)
    
                return results

                #return results
            return wrap_func
        return decorator_fn

    def register_group_by(self, func):
        self.group_by = func
        return func
    def register_reduce(self, func):
        self.reduce_fn = func
        return func

distributed = Distributed(num_workers=5, max_size=100)