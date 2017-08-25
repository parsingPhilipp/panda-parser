import subprocess


def render_and_view_dog(dog, name, path="/tmp/"):
    dot = dog.export_dot(name)
    dot_path = path + name + '.dot'
    pdf_path = path + name + '.pdf'
    with open(dot_path, 'w') as dot_file:
        dot_file.write(dot)

    command = ["dot", "-Tpdf", dot_path, "-o", pdf_path]
    p = subprocess.Popen(command)
    p.communicate()
    # print(command, p.returncode, p.stderr, p.stdout)

    q = subprocess.Popen(["zathura", pdf_path])
    return q
