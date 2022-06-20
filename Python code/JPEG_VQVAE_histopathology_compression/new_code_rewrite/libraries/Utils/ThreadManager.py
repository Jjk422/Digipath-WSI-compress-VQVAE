# class ThreadManager():
#     threads = []
#     for batch_index, (images, classification) in enumerate(dataset.get_test_loader()):
#         threads.append(Thread(target=self.__test_batch, args=(images, batch_index, dataset)))
#         threads[batch_index].start()
#
#     progress_bar = ProgressBar(len(dataset.get_test_loader()))
#     for thread in threads:
#         thread.join()
#         progress_bar.update(1)
#
#     progress_bar.close()

from concurrent.futures.thread import ThreadPoolExecutor
from tqdm import tqdm
# from tqdm.notebook import tqdm
from new_code_rewrite.libraries.Utils.ProgressBar import ProgressBar


class ThreadManager():
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish(wait=True)

    def __init__(self, max_workers=None, progress_bar=None):
        self.threadPool = ThreadPoolExecutor() if max_workers is None else ThreadPoolExecutor(max_workers=max_workers)
        self.progress_bar = ProgressBar() if progress_bar is None else progress_bar

    def add_thread(self, function, *args, **kwargs):
        self.threadPool.submit(function, *args, **kwargs)

    def finish(self, wait=True):
        self.threadPool.shutdown(wait=wait)

def ThreadManagerTQDM(function, argument_iterator, progress_bar_description, display_progress_bar=True, progress_bar_position=0):
    with ThreadPoolExecutor() as executor:
        if display_progress_bar:
            results = list(tqdm(executor.map(function, argument_iterator), total=len(argument_iterator), desc=progress_bar_description, position=progress_bar_position))
        else:
            results = list(executor.map(function, argument_iterator))
    return results