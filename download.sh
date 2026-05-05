#!/bin/bash

# Download the WeedMap RoWeeder-processed dataset
gdown 1NfkNXJvRNnabFK6CTFb5RaEiFEHkjvYk

# Unzip the dataset
unzip WeedMap_RoWeeder.zip -d .

# Remove the zip file
rm WeedMap_RoWeeder.zip