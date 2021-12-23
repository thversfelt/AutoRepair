# AutoRepair

Automated Repair for Autonomous Driving Systems.

## Getting Started

These instructions will cover usage information for AutoRepair. 

### Prerequisities

In order to run this container you'll need docker installed.

* [Windows](https://docs.docker.com/windows/started)
* [OS X](https://docs.docker.com/mac/started/)
* [Linux](https://docs.docker.com/linux/started/)

### Usage

To download and run AutoRepair in a docker container, run the following command in a terminal.

```shell
docker run thversfelt/autorepair:latest
```

Currently, it will run my python implementation of ARIEL on a simple faulty rule set that controls a Self-Driving Vehicle in several scenarios of highway-env. It will often take several generations to find an improved rule set, sometimes it finds an improved rule set in a single generation, and occasionally it will get stuck or crash due to an obscure recursion bug.
