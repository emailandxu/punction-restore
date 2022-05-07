# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import load_yaml
from util.utils import preprocess_paths


class Config:
    """ User configs class for training, testing or infering """

    def __init__(self, path: str):
        print('configs file path:', path)
        config = load_yaml(preprocess_paths(path))
        self.vocab_config = config.get("vocab_config", {})
        self.model_config = config.get("model_config", {})
        self.dataset_config = config.get("dataset_config", {})
        self.optimizer_config = config.get("optimizer_config", {})
        self.running_config = config.get("running_config", {})
    def __repr__(self):
        return str(self)

    def __str__(self):
        string = ''
        string += '#==================================================' + '\n'
        string += '#speech config: ' + str(self.vocab_config) + '\n'
        string += '#--------------------------------------------------' + '\n'
        string += '#model config: ' + str(self.model_config) + '\n'
        string += '#--------------------------------------------------' + '\n'
        string += '#dataset config: ' + str(self.dataset_config) + '\n'
        string += '#--------------------------------------------------' + '\n'
        string += '#optimizer config: ' + str(self.optimizer_config) + '\n'
        string += '#--------------------------------------------------' + '\n'
        string += '#running config: ' + str(self.running_config) + '\n'
        string += '#==================================================' + '\n'
        return string
