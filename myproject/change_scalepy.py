import re
import datetime

def scale_field(text, field, k=10):
    pattern = rf'{field}="([\d.-]+) ([\d.-]+) ([\d.-]+)"'
    def replace(match):
        x, y, z = match.groups()
        x = str(float(x) * k)
        y = str(float(y) * k)
        z = str(float(z) * k)
        print(f'{field}="{x} {y} {z}"')
        return f'{field}="{x} {y} {z}"'

    return re.sub(pattern, replace, text)

def scale_size_1(text, k=10):
    pattern = r'size="([\d.-]+)"'
    def replace(match):
        x = match.group(1)
        x = str(float(x) * k)
        return f'size="{x}"'

    return re.sub(pattern, replace, text)

def scale(text, k=10):
    text = scale_field(text, 'pos', k)
    text = scale_field(text, 'size', k)
    text = scale_field(text, 'scale', k)
    text = scale_size_1(text, k)
    return text

if __name__ == '__main__':
    with open('jaco2.xml') as f:
        text = f.read()
    k=5
    text = scale(text, k)

    with open(f'jaco2_{k}.xml', 'w') as f:
        f.write(text)