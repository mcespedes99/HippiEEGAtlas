import re

pattern = r"sub-P(\d{3})$"

if re.match(pattern, "sub-P002"):
    print('true')
else:
    print('false')