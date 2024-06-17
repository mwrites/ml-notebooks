#!/bin/bash

# Create a temporary file listing all the input videos
for f in lunar_lander_episode_*.mp4; do
  echo "file '$f'" >> temp_list.txt
done

# Sort the list numerically
sort -V temp_list.txt > sorted_list.txt

# Merge the videos
ffmpeg -f concat -safe 0 -i sorted_list.txt -c copy merged_lunar_lander.mp4

# Clean up temporary files
rm temp_list.txt sorted_list.txt
