import pandas as pd
import os
from datetime import datetime
import asyncio

class run_log:
    def __init__(self, model_info: pd.DataFrame, filepath: str = "./data/run"):
        self.it = 0

        self.filepath = f"{filepath}/{datetime.now().strftime("%d_%H_%M")}"
        if not os.path.isdir(self.filepath):
            os.mkdir(self.filepath)
        model_info.to_csv(f"{self.filepath}/overview.txt", "\t")
    
    def write_gradients(self, gradients: list):
        for index, gradient in enumerate(gradients):
            asyncio.run(self.write_gradient(f"{self.filepath}/gradient{index}.txt", gradient))
        self.it+=1
    
    async def write_gradient(self, filename, gradient):
        with open(filename, "a") as file:
            file.write(f"iteration {self.it}")
            for row in gradient:
                file.write(str(row))
                
            