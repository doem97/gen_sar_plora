#!/bin/bash

find . -type l -print0 | while read -d $'\0' link; do
  echo "$link"
done
