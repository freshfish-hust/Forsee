import sys
import json


sys.path.append(r'./')

from codesecurity.feature.objects import CommonFeatureSet

test_file='dataset/js-test/non-vul/3n3m1%40ukr.net__coderaiser%2Fcloudcmd__client.js__6c263c8c02ccf6a7fd9c51b1d8c303594f9ee6ac.js'


feature=CommonFeatureSet.from_file(test_file)


print(feature.tokens)