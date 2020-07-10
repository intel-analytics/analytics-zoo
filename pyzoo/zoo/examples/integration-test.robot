*** Settings ***
Documentation    Zoo Python Integration Test
Resource         integration-test.robot
Suite Setup      Prepare DataSource And Verticals
Suite Teardown   Delete All Sessions
Test template    Zoo Test

*** Variables ***
@{verticals}  ${spark_210_3_vid}    ${hdfs_264_3_vid}

*** Test Cases ***   SuiteName                             VerticalId
1                    Yarn Test Suite                       ${hdfs_264_3_vid}

*** Keywords ***
Build SparkJar
   [Arguments]       ${spark_profile}
   ${build}=         Catenate                        SEPARATOR=/    ${curdir}    pyzoo/dev/release.sh linux default false
   Log To Console    ${spark_profile}
   Log To Console    start to build jar
   Log To Console    pyzoo/dev/release.sh linux default false -P ${spark_profile} -Dbigdl.version=${bigdl_version}...
   Remove Directory  pyzoo/dist/
   Run               ${build} -P ${spark_profile} -Dbigdl.version=${bigdl_version}
   {zoo_whl}=         Catenate                        SEPARATOR=/    ${curdir}    analytics_zoo-*.dev0-py2.py3-none-manylinux1_x86_64.whl
   Log To Console    build jar finished

Yarn Test Suite
   Log To Console                   Start the Yarn Test Suite
   Build SparkJar                   spark_2.x
   Set Environment Variable         SPARK_HOME               /opt/work/spark-2.1.0-bin-hadoop2.7
   Set Environment Variable         HADOOP_CONF_DIR          /opt/work/hadoop-2.6.5/etc/hadoop
   Set Environment Variable         http_proxy               ${http_proxy}
   Set Environment Variable         https_proxy              ${https_proxy}
   Set Environment Variable         ANALYTICS_ZOO_WHL        ${zoo_whl}
   RUN                              conda create -y -n horovod-pytorch python==3.6.7
   RUN                              conda env remove -y -n horovod-pytorch
   Log To Console                   success
