from datetime import datetime


class app_logger:
    def __init__(self):
        pass

    def log(self, file_obj, log_message):
        self.now = datetime.now()
        self.date = self.now.date()
        self.curr_time = self.now.strftime("%H:%M:%S")
        file_obj.write(
            str(self.date) + "/" + str(self.curr_time) + "\t\t" + log_message + "\n"
        )
