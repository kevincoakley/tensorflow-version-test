#!/usr/bin/env python

import yaml
from jinja2 import Environment, FileSystemLoader

# Read the variables for the tests
with open("test_versions.yml") as f:
    test_versions = yaml.safe_load(f)

# Use a jinja template to create the Singularity defintion files
env = Environment(loader=FileSystemLoader("templates"), trim_blocks=True,  lstrip_blocks=True)
template = env.get_template("singularity.def.j2")

for test_variables in test_versions:
    parsed_template = template.render(test_variables)
 
    with open("centos-cpu-%s-tf%s.def" % (test_variables["centos_version"],
    test_variables["tensorflow_version"]), "w") as fh:
        fh.write(parsed_template)


# Use a jinja template to create the shell script to run the tests
template = env.get_template("build_singularity.sh.j2")
parsed_template = template.render(test_versions=test_versions)

with open("build_singularity.sh", "w") as fh:
    fh.write(parsed_template)


# Use a jinja template to create the shell script to run the tests
template = env.get_template("run_singularity_tests.sh.j2")
parsed_template = template.render(test_versions=test_versions)

with open("run_singularity_tests.sh", "w") as fh:
    fh.write(parsed_template)
