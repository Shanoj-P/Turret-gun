#!/usr/bin/env python
import rosbag
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python bag_to_csv.py <rosbag_file>")
    sys.exit(1)

bag_file = sys.argv[1]

print(f"Processing {bag_file}...")

bag = rosbag.Bag(bag_file)
topics = bag.get_type_and_topic_info()[1].keys()
data = {}

for topic in topics:
    data[topic] = []

for topic, msg, t in bag.read_messages():
    msg_dict = {}
    msg_dict["timestamp"] = t.to_sec()

    if hasattr(msg, "__slots__"):
        for field in msg.__slots__:
            msg_dict[field] = getattr(msg, field)

    data[topic].append(msg_dict)

bag.close()

for topic, records in data.items():
    if records:
        df = pd.DataFrame(records)
        csv_filename = topic.replace("/", "_") + ".csv"
        df.to_csv(csv_filename, index=False)
        print(f"Saved: {csv_filename}")

print("Conversion complete!")
