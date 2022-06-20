from tqdm import tqdm
# from tqdm.notebook import tqdm

class ProgressBar():
    # TODO: Find a better way to stop progress bar displaying
    def __init__(self, total_value=None, description=None, position=0, leave=True, display_progress_bar=True):
        self.display_progress_bar = display_progress_bar
        if self.display_progress_bar:
            self.progressBar = tqdm(total=total_value, desc=description, position=position, leave=leave) if total_value is not None else tqdm(desc=description, position=position, leave=leave)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if self.display_progress_bar:
            self.progressBar.close()

    def update(self, update_value=1):
        if self.display_progress_bar:
            self.progressBar.update(update_value)

    def reset(self, reset_value=0):
        if self.display_progress_bar:
            self.progressBar.n = reset_value
            self.progressBar.refresh()
        # self.reset(reset_value)

    def close(self):
        if self.display_progress_bar:
            self.progressBar.close()