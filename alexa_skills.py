#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Start Tagged Instances on AWS using Lambda / Alexa
#
#
#

from __future__ import print_function
from __future__ import unicode_literals

import boto3
import logging
import os

# setup simple logging for INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# CONSTANTS
ALEXA_AWS_DEFAULT_TAG = 'alexa'
AWS_DEFAULT_REGION = 'North Virginia'

# Region Mapping
AWS_REGION_MAPPING = {
    'North Virginia': 'us-east-1',
    'Ohio': 'us-east-2',
    'North California': 'us-west-1',
    'Oregon': 'us-west-2',
    'Canada': 'ca-central-1',
    'Ireland': 'eu-west-1',
    'Frankfurt': 'eu-central-1',
    'London': 'eu-west-2',
    'Singapore': 'ap-southeast-1',
    'Sydney': 'ap-southeast-2',
    'Seoul': 'ap-northeast-2',
    'Tokyo': 'ap-northeast-1',
    'Mumbai': 'ap-south-1',
    'Sao Paulo': 'sa-east-1'
}


def lambda_handler(event, context):
    
    print("event.session.application.applicationId=" +
          event['session']['application']['applicationId'])

    

    if event['session']['new']:
        on_session_started({'requestId': event['request']['requestId']}, event['session'])

    if event['request']['type'] == "LaunchRequest":
        return on_launch(event['request'], event['session'])

    elif event['request']['type'] == "IntentRequest":
        return on_intent(event['request'], event['session'])

    elif event['request']['type'] == "SessionEndedRequest":
        return on_session_ended(event['request'], event['session'])


def on_session_started(session_started_request, session):
    """
    Called when the session starts
    """

    print("on_session_started requestId=" + session_started_request['requestId']
          + ", sessionId=" + session['sessionId'])


def on_launch(launch_request, session):
    """
    Called when the user launches the skill without specifying what they want
     
    """

    print("on_launch requestId=" + launch_request['requestId'] +
          ", sessionId=" + session['sessionId'])

    # Welcome on skill launch
    session_attributes = {}
    reprompt_text = None
    should_end_session = False
    card_title = "Welcome"

    speech_output = "Welcome to Alexa Voice Operations for AWS."

    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))


def on_intent(intent_request, session):
    """
    Called when the user specifies an intent for this skill 
   
    """

    print("on_intent requestId=" + intent_request['requestId'] +
          ", sessionId=" + session['sessionId'])

    intent = intent_request['intent']
    intent_name = intent_request['intent']['name']

    # Dispatch to your skill's intent handlers
    if intent_name == "StartTaggedInstances":
        return start_tagged_instances(intent, session)
	
    elif intent_name == "StartNonTaggedInstances":
	    return start_non_tagged_instances(intent, session)

    elif intent_name == "StopTaggedInstances":
        return stop_tagged_instances(intent, session)
	
    elif intent_name == "StopNonTaggedInstances":
        return stop_non_tagged_instances(intent, session)

    elif intent_name == "GetRunningTaggedInstances":
        return get_running_tagged_instances(intent, session)
		
    elif intent_name == "GetRunningNonTaggedInstances":
        return get_running_non_tagged_instances(intent, session)

    elif intent_name == "GetCurrentRegion":
        return get_current_region(intent, session)

    elif intent_name == "SetCurrentRegion":
        return set_current_region(intent, session)
    
    elif intent_name == "LaunchInstance":
        return get_instance_setup(intent, session)

    elif intent_name == "AMAZON.HelpIntent":
        return get_skill_help_response()

    else:
        raise ValueError("Invalid intent")


def on_session_ended(session_ended_request, session):
    """
    Called when the user ends the session.
    Is not called when the skill returns should_end_session=true
    
    """
    print("on_session_ended requestId=" + session_ended_request['requestId'] +
          ", sessionId=" + session['sessionId'])

# --------------- Functions that control the skill's behavior ------------------

def get_instance_setup(intent, session):
    session_attributes = {}
    reprompt_text = None
    should_end_session = True

    # get current region
    region_name = get_region_from_session_helper(session)
    # define the connection
    ec2 = boto3.resource('ec2', region_name=region_name)
    instance = ec2.create_instances(
        ImageId = 'ami-0323c3dd2da7fb37d',
        InstanceType = 't2.micro',
        KeyName = 'bashu1602',
        MaxCount=1,
        MinCount=1)
    filters = [
        {
            'Name': 'instance-state-name',
            'Values': ['running']
        }
    ]
	# Start the filtering
    instances = ec2.instances.filter(Filters=filters)

    # Get all the instances that are running
    running_instances = [instance.id for instance in instances]

    # print these for the log
    print(running_instances)

    # speed output for the user
    speech_output = "EC2 instance is launched. Please check the console."

    # Setting reprompt_text to None signifies that we do not want to reprompt
    # the user.
    return build_response(session_attributes, build_speechlet_response(
        intent['name'], speech_output, reprompt_text, should_end_session))


def stop_tagged_instances(intent, session):
    """
    Stop Running instances with the Tag ALEXA_AWS_DEFAULT_TAG 
    """

    session_attributes = {}
    reprompt_text = None
    should_end_session = False

    # get current region
    region_name = get_region_from_session_helper(session)

    # define the connection
    ec2 = boto3.resource('ec2', region_name=region_name)

    # Filter for Instances that are running
    # and have ALEXA_AWS_DEFAULT_TAG with the Value `True`
    filters = [
        {
            'Name': 'tag:' + ALEXA_AWS_DEFAULT_TAG,
            'Values': ['True']
        },
        {
            'Name': 'instance-state-name',
            'Values': ['running']
        }
    ]

    # Start the filtering
    instances = ec2.instances.filter(Filters=filters)

    # Get all the instances that are running
    running_instances = [instance.id for instance in instances]

    # print these for the log
    print(running_instances)

    # check that we have some stopped instances
    if len(running_instances) > 0:
        print("Stopping now running Instances")

        ec2.instances.filter(InstanceIds=running_instances).stop()

    else:
        print("Could not find any running Instance")

    # speed output for the user
    speech_output = "We stopped {} tagged Instances for you".format(str(len(running_instances)))

    # Setting reprompt_text to None signifies that we do not want to reprompt
    # the user. 
    return build_response(session_attributes, build_speechlet_response(
        intent['name'], speech_output, reprompt_text, should_end_session))

def stop_non_tagged_instances(intent, session):
    """
    Stop Running non tagged instances 
    """

    session_attributes = {}
    reprompt_text = None
    should_end_session = False

    # get current region
    region_name = get_region_from_session_helper(session)

    # define the connection
    ec2 = boto3.resource('ec2', region_name=region_name)

    # Filter for non tagged Instances that are running
    filters = [
        {
            'Name': 'tag:' + ALEXA_AWS_DEFAULT_TAG,
            'Values': ['True']
        },
        {
            'Name': 'instance-state-name',
            'Values': ['running']
        }
    ]
    filters_2 = [
	    
        {
            'Name': 'instance-state-name',
            'Values': ['running']
        }
    ]


    # Start the filtering
    instances = ec2.instances.filter(Filters=filters)
    instances_2 = ec2.instances.filter(Filters=filters_2)

    # Get all the instances that are running
    running_instances = [instance.id for instance in instances]
    running_instances_2 = [instance.id for instance in instances_2]
    instances_to_stop = [to_stop for to_stop in running_instances_2 if to_stop not in running_instances]

    # print these for the log
    print(instances_to_stop)

    # check that we have some stopped instances
    if len(instances_to_stop) > 0:
        print("Stopping now running untagged Instances")

        ec2.instances.filter(InstanceIds=instances_to_stop).stop()

    else:
        print("Could not find any running untagged Instance")

    # speed output for the user
    speech_output = "We stopped {} untagged Instances for you".format(str(len(instances_to_stop)))

    # Setting reprompt_text to None signifies that we do not want to reprompt
    # the user. 
    return build_response(session_attributes, build_speechlet_response(
        intent['name'], speech_output, reprompt_text, should_end_session))
		
def start_tagged_instances(intent, session):
    """
    Start stopped instances with the Tag ALEXA_AWS_DEFAULT_TAG 
    
    """

    session_attributes = {}
    reprompt_text = None
    should_end_session = False

    # get current region
    region_name = get_region_from_session_helper(session)

    # define the connection
    ec2 = boto3.resource('ec2', region_name=region_name)

    # Filter for Instances that are stopped
    # and have ALEXA_AWS_DEFAULT_TAG with the Value `True`
    filters = [
        {
            'Name': 'tag:' + ALEXA_AWS_DEFAULT_TAG,
            'Values': ['True']
        },
        {
            'Name': 'instance-state-name',
            'Values': ['stopped']
        }
    ]

    # Start the filtering
    instances = ec2.instances.filter(Filters=filters)

    # Get all the instances that are off
    stopped_instances = [instance.id for instance in instances]

    # print these for the log
    print(stopped_instances)

    # check that we have some stopped instances
    if len(stopped_instances) > 0:
        print("Starting now stopped Instances")

        ec2.instances.filter(InstanceIds=stopped_instances).start()

    else:

        print("Could not find any stopped Instance")

    # speed output for the user
    speech_output = "We started {} tagged Instances for you".format(str(len(stopped_instances)))

    # Setting reprompt_text to None signifies that we do not want to reprompt
    # the user.
    return build_response(session_attributes, build_speechlet_response(
        intent['name'], speech_output, reprompt_text, should_end_session))
		
def start_non_tagged_instances(intent, session):
    """
    Start stopped non tagged instances  
    On the current Region Set
    """

    session_attributes = {}
    reprompt_text = None
    should_end_session = False

    # get current region
    region_name = get_region_from_session_helper(session)

    # define the connection
    ec2 = boto3.resource('ec2', region_name=region_name)

    # Filter for all non tagged Instances that are stopped
    
    filters = [
	    {
            'Name': 'tag:' + ALEXA_AWS_DEFAULT_TAG,
            'Values': ['True']
        },
        {
            'Name': 'instance-state-name',
            'Values': ['stopped']
        }
    ]
	
    filters_2 = [
	    
        {
            'Name': 'instance-state-name',
            'Values': ['stopped']
        }
    ]

    # Start the filtering
    instances = ec2.instances.filter(Filters=filters)
    instances_2 = ec2.instances.filter(Filters=filters_2)
	
    # Get all the instances that are off
    stopped_instances = [instance.id for instance in instances]
    stopped_instances_2 = [instance.id for instance in instances_2]
    instances_to_start = [to_start for to_start in stopped_instances_2 if to_start not in stopped_instances]

    # print these for the log
    print(instances_to_start)

    # check that we have some stopped instances
    if len(instances_to_start) > 0:
        print("Starting now stopped untagged Instances")

        ec2.instances.filter(InstanceIds=instances_to_start).start()

    else:

        print("Could not find any stopped untagged Instance")

    # speed output for the user
    speech_output = "We started {} untagged Instances for you".format(str(len(instances_to_start)))

    # Setting reprompt_text to None signifies that we do not want to reprompt
    # the user. 
    return build_response(session_attributes, build_speechlet_response(
        intent['name'], speech_output, reprompt_text, should_end_session))


def get_running_tagged_instances(intent, session):
    """
    Tell the user how many instances we have running
  
    """

    session_attributes = {}
    reprompt_text = None
    should_end_session = False

    # get current region
    region_name = get_region_from_session_helper(session)

    # define the connection
    ec2 = boto3.resource('ec2', region_name=region_name)

    # Filter for Instances that are running
    # and have ALEXA_AWS_DEFAULT_TAG with the Value `True`
    filters = [
        {
            'Name': 'tag:' + ALEXA_AWS_DEFAULT_TAG,
            'Values': ['True']
        },
        {
            'Name': 'instance-state-name',
            'Values': ['running']
        }
    ]
    filters_2 = [
	    
        {
            'Name': 'instance-state-name',
            'Values': ['running']
        }
    ]

    # Start the filtering
    instances = ec2.instances.filter(Filters=filters)

    # Get all the instances that are running
    running_instances = [instance.id for instance in instances]

    # print these for the log
    print(running_instances)

    # speed output for the user
    speech_output = "Currently, there are {} running tagged Instances.".format(str(len(running_instances)))

    # Setting reprompt_text to None signifies that we do not want to reprompt
    # the user.
    return build_response(session_attributes, build_speechlet_response(
        intent['name'], speech_output, reprompt_text, should_end_session))
		
def get_running_non_tagged_instances(intent, session):
    """
    Tell the user how many non tagged instances we have running
    
    """

    session_attributes = {}
    reprompt_text = None
    should_end_session = False

    # get current region
    region_name = get_region_from_session_helper(session)

    # define the connection
    ec2 = boto3.resource('ec2', region_name=region_name)

    # Filter for Instances that are running
    # and have ALEXA_AWS_DEFAULT_TAG with the Value `True`
    filters = [
        {
            'Name': 'tag:' + ALEXA_AWS_DEFAULT_TAG,
            'Values': ['True']
        },
        {
            'Name': 'instance-state-name',
            'Values': ['running']
        }
    ]
    filters_2 = [
	    
        {
            'Name': 'instance-state-name',
            'Values': ['running']
        }
    ]

    # Start the filtering
    instances = ec2.instances.filter(Filters=filters)
    instances_2 = ec2.instances.filter(Filters=filters_2)
	
    # Get all the instances that are running
    running_instances = [instance.id for instance in instances]
    running_instances_2 = [instance.id for instance in instances_2]
    instances_currently_running = [to_stop for to_stop in running_instances_2 if to_stop not in running_instances]

    # print these for the log
    print(instances_currently_running)

    # speed output for the user
    speech_output = "Currently, there are {} running untagged Instances.".format(str(len(instances_currently_running)))

    # Setting reprompt_text to None signifies that we do not want to reprompt
    # the user. 
    return build_response(session_attributes, build_speechlet_response(
        intent['name'], speech_output, reprompt_text, should_end_session))
		


def get_current_region(intent, session):
    """
    Gets the current Region
    
    """

    session_attributes = {}
    reprompt_text = None
    should_end_session = False

    # CASE: Region is set
    if session.get('attributes', {}) and "currentRegion" in session.get('attributes', {}):
        aws_region = session['attributes']['currentRegion']
        session_attributes = {"currentRegion": aws_region}
        speech_output = "Your current AWS region is " + aws_region
       
    # CASE: There was not region set for the session
    else:
        speech_output = "You did not set any Region. " \
                        "We are using therefore the default Region " + AWS_DEFAULT_REGION + \
                        ". Please tell me your current AWS Region by saying, " \
                        "Set my region to North Virginia."

    # Setting reprompt_text to None signifies that we do not want to reprompt
    # the user. 
    return build_response(session_attributes, build_speechlet_response(
        intent['name'], speech_output, reprompt_text, should_end_session))


def set_current_region(intent, session):
    """
    Sets the current AWS Region we are working on
    
    """

    card_title = intent['name']
    session_attributes = {}
    should_end_session = False

    # CASE: The Region provided by the user was found in our slots
    if 'Region' in intent['slots']:

        aws_region = intent['slots']['Region']['value']
        session_attributes = create_aws_region_attribute(aws_region)

        speech_output = "The AWS Region is now set to " + \
                        aws_region + \
                        ". You can ask me for the current AWS Region by saying, " \
                        "what is the current Region?"

        reprompt_text = "You can ask me for the current AWS Region by saying, " \
                        "what is the current Region?"

    # CASE: The region could not be found in our slots§§§§§§§
    else:
        speech_output = "I am not sure what AWS region you need. " \
                        "Please try again."

        reprompt_text = "I am not sure what AWS region you need. " \
                        "Please tell me your current AWS Region by saying," \
                        "Set my region to North Virginia."

    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))


def get_skill_help_response():
    """
    If we wanted to initialize the session to have some attributes we could
    add those here
    
    """

    session_attributes = {}
    card_title = "Welcome"
    should_end_session = False

    speech_output = "Welcome to the AWS Management Skill. " \
                    "Please tell me the current region you want to work with by saying," \
                    "Set my region to North Virginia."

    # If the user either does not reply to the welcome message or says something
    # that is not understood, they will be prompted again with this text.
    reprompt_text = "Please tell me your current AWS Region by saying," \
                    "Set my region to North Virginia."

    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))


# --------------- Helpers that build all of the responses ----------------------


def create_aws_region_attribute(region):
    """
    Returns the structure for the currentRegion
    """
    return {"currentRegion": region}


def get_region_from_session_helper(session):
    """
    Returns the current session region or the default

    """
    # CASE: Region is set
    if session.get('attributes', {}) and "currentRegion" in session.get('attributes', {}):
        aws_region_full = session['attributes']['currentRegion']

        if aws_region_full.strip() in AWS_REGION_MAPPING.keys():

            return AWS_REGION_MAPPING[aws_region_full.strip()]

        else:

            return AWS_REGION_MAPPING[AWS_DEFAULT_REGION]

    else:

        return AWS_REGION_MAPPING[AWS_DEFAULT_REGION]


def build_speechlet_response(title, output, reprompt_text, should_end_session):
    """
    Build Speechlet Response
    
    :param title: 
    :param output: 
    :param reprompt_text: 
    :param should_end_session: 
    :return: 
    """
    return {
        'outputSpeech': {
            'type': 'PlainText',
            'text': output
        },
        'card': {
            'type': 'Simple',
            'title': 'SessionSpeechlet - ' + title,
            'content': 'SessionSpeechlet - ' + output
        },
        'reprompt': {
            'outputSpeech': {
                'type': 'PlainText',
                'text': reprompt_text
            }
        },
        'shouldEndSession': should_end_session
    }


def build_response(session_attributes, speechlet_response):
    """
    Build the Response
    
    :param session_attributes: 
    :param speechlet_response: 
    :return: 
    """
    return {
        'version': '1.0',
        'sessionAttributes': session_attributes,
        'response': speechlet_response
    }