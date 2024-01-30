# This is
#
#
#SPECIFIC to the GNU/Linux Bourne shell,
#
#
# and different
# shells and/or OSs will have slightly different syntax, but
# essentially this one-line scripts prepend the folder that our
# package lives in to the $PYTHONPATH environmental variable,
# so that we can import modules from our package from an
# arbitrary location in the filesystem---not only the cwd.
#
#
# UNDER LINUX DO
# 'source env.sh'
# in the terminal and you're ready to go.
#
# Adapt this to your OS and filesystem.
#

# See this stackoverflow question
# http://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
# for the magic in this command
SETUP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=$SETUP_DIR:$PYTHONPATH
