# Environment Setup

Sometimes, the most frustrating thing to do during onboarding is getting your developer environment to work. I'll try to make as streamlined as possible in this section.

There are a few ways to install the necessary Python libraries to run the different repos I have in Lambda. However, were it up to me, I would just copy and paste the Python
environment that I currently have and run from its bin.

Here's how to do it:

First, copy my python\_env directory, which contains all the necessary libraries with this command: 

```md
cp -r /data/aneurysm/tranch29/python_env [destination-path]
```

To execute the python command with all the required libraries, you would then need to call on the python binary in python_\env. 
Say you copied it to your home directory. To use Python on a script named helloworld.py in your current directory, you would run:

```md
~/python_env/bin/python helloworld.py
```
However, nobody wants to type the entire path every time they want to run a python path. I HIGHLY recommend [setting up an alias](https://www.tecmint.com/create-alias-in-linux/).
