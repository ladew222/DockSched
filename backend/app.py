from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS, cross_origin
import pulp
import re
import csv
import json
from io import StringIO
from urllib.parse import urlencode
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
import calendar
app = Flask(__name__)
import hashlib
import pandas as pd
import random
import numpy as np
import random
from datetime import datetime, timedelta
from collections import Counter
import uuid
import os
from flask import make_response
from ics import Calendar, Event
from datetime import datetime, timedelta
import re
import pandas as pd
from io import StringIO
from pulp import *
from flask import Flask
from flask_cors import CORS
#from gurobipy import Model, GRB, quicksum
from deap import base, creator, tools, algorithms
import io

app = Flask(__name__)
CORS(app)



global_settings = {}
results_storage = {}

# Define the problem class
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def string_to_color(s):
    # Use a hash function to convert the string to a hexadecimal color code
    hash_object = hashlib.md5(s.encode())
    return '#' + hash_object.hexdigest()[:6]


def follows_pattern(timeslot, pattern):
    """Check if a timeslot follows a specific pattern (MWF or TuTh)."""
    days = timeslot.split(' - ')[0]
    if pattern == 'MWF':
        return all(day in days for day in ['M', 'W', 'F'])
    elif pattern == 'TuTh':
        return all(day in days for day in ['Tu', 'Th'])
    return False


def create_individual(failed_sections,combined_expanded_schedule, mutation_rate=0.3):
    individual = []
    day_assignment = {}  # Dictionary to track day assignments for each course
    full_meeting_times = create_full_meeting_times()

    for cls in combined_expanded_schedule:
        # Copy class section detailså
        new_class_section = cls.copy()
        course_identifier = cls['section'].split('_')[0]
        
        
        first_elements_list = [item[0] for item in failed_sections]
        # Check if the current class section is in the list of failed sections
        if course_identifier in first_elements_list or 1==1:
            # Randomly decide whether to mutate this class section
            if random.random() < mutation_rate:
                # Determine the current pattern (MWF or TuTh) of the class
                current_pattern = 'MWF' if 'M' in cls['timeslot'] or 'W' in cls['timeslot'] or 'F' in cls['timeslot'] else 'TuTh'
                
                # Filter timeslots based on the pattern and avoid clashes on the same day
                available_timeslots = [
                    ts for ts in full_meeting_times
                    if ts['days'] in cls['timeslot'] and
                       ts['days'] not in day_assignment.get(course_identifier, set())
                ]

                # If available timeslots are found, choose one randomly
                if available_timeslots:
                    chosen_timeslot = random.choice(available_timeslots)
                    new_class_section['timeslot'] = f"{chosen_timeslot['days']} - {chosen_timeslot['start_time']}"
                    # Update day assignment for the course
                    day_assignment.setdefault(course_identifier, set()).add(chosen_timeslot['days'])

        # Add the class section to the individual schedule regardless of mutation
        individual.append(new_class_section)

    return creator.Individual(individual)





def create_meeting_times():
        your_meeting_time_data = [
        {
            'days': 'M W F',
            'start_time': '8:00AM',
            'end_time': '8:55AM',
        },
        {
            'days': 'M W F',
            'start_time': '9:05AM',
            'end_time': '10:00AM',
        },
        {
            'days': 'M W F',
            'start_time': '10:10AM',
            'end_time': '11:05AM',
        },
        {
            'days': 'M W F',
            'start_time': '11:15AM',
            'end_time': '12:10PM',
        },
        {
            'days': 'M W F',
            'start_time': '12:20PM',
            'end_time': '1:15PM',
        },
        {
            'days': 'M W F',
            'start_time': '1:25PM',
            'end_time': '2:20PM',
        },
        {
            'days': 'M W F',
            'start_time': '2:30PM',
            'end_time': '3:25PM',
        },
        {
            'days': 'M W F',
            'start_time': '3:35PM',
            'end_time': '4:30PM',
        },
        {
            'days': 'Tu Th',
            'start_time': '8:00AM',
            'end_time': '9:20AM',
        },
        {
            'days': 'Tu Th',
            'start_time': '9:30AM',
            'end_time': '10:50AM',
        },
        {
            'days': 'Tu Th',
            'start_time': '11:00AM',
            'end_time': '12:20PM',
        },
        {
            'days': 'Tu Th',
            'start_time': '12:30PM',
            'end_time': '1:50PM',
        },
        {
            'days': 'Tu Th',
            'start_time': '2:00PM',
            'end_time': '3:20PM',
        },
        {
            'days': 'Tu Th',
            'start_time': '3:30PM',
            'end_time': '4:50PM',
        },
    ]

        return your_meeting_time_data
    

def create_full_meeting_times():
    your_meeting_time_data = [
        {
            'days': 'M',
            'start_time': '8:00AM',
            'end_time': '8:55AM',
        },
        {
            'days': 'M',
            'start_time': '9:05AM',
            'end_time': '10:00AM',
        },
        {
            'days': 'M',
            'start_time': '10:10AM',
            'end_time': '11:05AM',
        },
        {
            'days': 'M',
            'start_time': '11:15AM',
            'end_time': '12:10PM',
        },
        {
            'days': 'M',
            'start_time': '12:20PM',
            'end_time': '1:15PM',
        },
        {
            'days': 'M',
            'start_time': '1:25PM',
            'end_time': '2:20PM',
        },
        {
            'days': 'M',
            'start_time': '2:30PM',
            'end_time': '3:25PM',
        },
        {
            'days': 'M',
            'start_time': '3:35PM',
            'end_time': '4:30PM',
        },
        {
            'days': 'W',
            'start_time': '8:00AM',
            'end_time': '8:55AM',
        },
        {
            'days': 'W',
            'start_time': '9:05AM',
            'end_time': '10:00AM',
        },
        {
            'days': 'W',
            'start_time': '10:10AM',
            'end_time': '11:05AM',
        },
        {
            'days': 'W',
            'start_time': '11:15AM',
            'end_time': '12:10PM',
        },
        {
            'days': 'W',
            'start_time': '12:20PM',
            'end_time': '1:15PM',
        },
        {
            'days': 'W',
            'start_time': '1:25PM',
            'end_time': '2:20PM',
        },
        {
            'days': 'W',
            'start_time': '2:30PM',
            'end_time': '3:25PM',
        },
        {
            'days': 'W',
            'start_time': '3:35PM',
            'end_time': '4:30PM',
        },
        {
            'days': 'F',
            'start_time': '8:00AM',
            'end_time': '8:55AM',
        },
        {
            'days': 'F',
            'start_time': '9:05AM',
            'end_time': '10:00AM',
        },
        {
            'days': 'F',
            'start_time': '10:10AM',
            'end_time': '11:05AM',
        },
        {
            'days': 'F',
            'start_time': '11:15AM',
            'end_time': '12:10PM',
        },
        {
            'days': 'F',
            'start_time': '12:20PM',
            'end_time': '1:15PM',
        },
        {
            'days': 'F',
            'start_time': '1:25PM',
            'end_time': '2:20PM',
        },
        {
            'days': 'F',
            'start_time': '2:30PM',
            'end_time': '3:25PM',
        },
        {
            'days': 'F',
            'start_time': '3:35PM',
            'end_time': '4:30PM',
        },
        {
            'days': 'Tu',
            'start_time': '8:00AM',
            'end_time': '9:20AM',
        },
        {
            'days': 'Tu',
            'start_time': '9:30AM',
            'end_time': '10:50AM',
        },
        {
            'days': 'Tu',
            'start_time': '11:00AM',
            'end_time': '12:20PM',
        },
        {
            'days': 'Tu',
            'start_time': '12:30PM',
            'end_time': '1:50PM',
        },
        {
            'days': 'Tu',
            'start_time': '2:00PM',
            'end_time': '3:20PM',
        },
        {
            'days': 'Tu',
            'start_time': '3:30PM',
            'end_time': '4:50PM',
        },
        {
            'days': 'Th',
            'start_time': '8:00AM',
            'end_time': '9:20AM',
        },
        {
            'days': 'Th',
            'start_time': '9:30AM',
            'end_time': '10:50AM',
        },
        {
            'days': 'Th',
            'start_time': '11:00AM',
            'end_time': '12:20PM',
        },
        {
            'days': 'Th',
            'start_time': '12:30PM',
            'end_time': '1:50PM',
        },
        {
            'days': 'Th',
            'start_time': '2:00PM',
            'end_time': '3:20PM',
        },
        {
            'days': 'Th',
            'start_time': '3:30PM',
            'end_time': '4:50PM',
        },
    ]

    return your_meeting_time_data


@app.route('/upload_student_schedule', methods=['POST'])
def upload_student_schedule():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read the uploaded file into a DataFrame
        student_schedule_data = pd.read_csv(file)
        
        # Ensure the necessary columns exist
        expected_columns = ['StudentID', 'Class1', 'Class2', 'Class3', 'Class4']  # Add more if necessary
        missing_columns = set(expected_columns) - set(student_schedule_data.columns)
        
        if missing_columns:
            # If there are missing columns, return an error
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing_columns)}'
            }), 400

        # Process and store this data for optimization
        process_student_schedules(student_schedule_data)
        return jsonify({'message': 'Student schedules uploaded successfully'}), 200
    
    
def process_student_schedules(student_schedule_data):
    student_class_mapping = {}
    for index, row in student_schedule_data.iterrows():
        student_id = row['StudentID']
        classes = [row[col] for col in student_schedule_data.columns if col != 'StudentID' and not pd.isnull(row[col])]
        student_class_mapping[student_id] = classes
    
    # Store this mapping globally or in a file/database for later use in optimization
    global_settings['student_class_mapping'] = student_class_mapping

class ClassSection:
    def __init__(self, sec_name, title, minCredit, sec_cap, room, bldg, week_days, csm_start, csm_end, faculty1, holdValue=None, restrictions=None, blocked_time_slots=None, assigned_meeting_time_indices=None):
        self.section = sec_name
        self.title = title
        self.minCredit = minCredit
        self.secCap = sec_cap
        self.room = room
        self.bldg = bldg
        self.week_days = week_days
        self.csm_start = csm_start
        self.csm_end = csm_end
        self.faculty1 = faculty1
        self.holdValue = holdValue

        # List of classes to avoid
        self.avoid_classes = []

        # List of unwanted timeslots
        self.unwanted_timeslots = []

        if assigned_meeting_time_indices is None:
            # Initialize assigned meeting time indices as an empty list
            self.assigned_meeting_time_indices = []
        else:
            # Use the provided assigned meeting time indices
            self.assigned_meeting_time_indices = assigned_meeting_time_indices

        # Add restrictions and blocked_time_slots to the lists
        self.add_restrictions(restrictions)
        self.add_unwanted_timeslots(blocked_time_slots)

        # If assigned meeting time indices are not vided, calculate them
        if not assigned_meeting_time_indices:
            meeting_times_data = create_meeting_times()
            self.calculate_assigned_meeting_time_indices(meeting_times_data)


    def add_restrictions(self, restrictions):
        # Check if restrictions is not None, is a string, and is not empty before splitting
        if restrictions and isinstance(restrictions, str) and restrictions.strip():
            # Split and add restrictions to the avoid_classes list
            self.avoid_classes.extend(restrictions.split(';'))

    def add_unwanted_timeslots(self, blocked_time_slots):
        # Check if blocked_time_slots is not None, is a string, and is not empty before splitting
        if blocked_time_slots and isinstance(blocked_time_slots, str) and blocked_time_slots.strip():
            # Split and add blocked_time_slots to the unwanted_timeslots list
            self.unwanted_timeslots.extend(blocked_time_slots.split(';'))

    def calculate_assigned_meeting_time_indices(self, meeting_times_data):
        # Initialize assigned meeting time indices as an empty list
        assigned_meeting_time_indices = []

        # Iterate through meeting times data
        for index, meeting_time in enumerate(meeting_times_data):
            if set(meeting_time['days']).issubset(set(self.week_days)):
                # Check if class days contain all meeting days
                start_time = meeting_time['start_time']
                end_time = meeting_time['end_time']
                class_start_time = self.csm_start

                # Check if class start time falls within the meeting time slot
                if start_time <= class_start_time <= end_time:
                    assigned_meeting_time_indices.append(index)

        # Update the assigned meeting time indices
        self.assigned_meeting_time_indices = assigned_meeting_time_indices



        # Update the assigned meeting time indices
        self.assigned_meeting_time_indices = assigned_meeting_time_indices

    def to_dictionary(self):
        try:
            # Convert the attributes of the class instance to a dictionary
            result = {
                'section': getattr(self, 'section', None),
                'title': getattr(self, 'title', None),
                'minCredit': getattr(self, 'minCredit', None),
                'secCap': getattr(self, 'secCap', None),
                'room': getattr(self, 'room', None),
                'bldg': getattr(self, 'bldg', None),
                'week_days': getattr(self, 'week_days', None),
                'csm_start': getattr(self, 'csm_start', None),
                'csm_end': getattr(self, 'csm_end', None),
                'faculty1': getattr(self, 'faculty1', None),
                'holdValue': getattr(self, 'holdValue', None),
                'avoid_classes': getattr(self, 'avoid_classes', None),
                'restrictions': getattr(self, 'unwanted_timeslots', None),
            }

            # Convert complex types (e.g., other objects) to dictionaries if needed
            # Example for a hypothetical complex attribute that requires conversion:
            # if isinstance(self.some_complex_attribute, SomeClass):
            #     result['some_complex_attribute'] = self.some_complex_attribute.to_dictionary()

            return result
        except Exception as e:
            # Log the exception or handle it as needed
            print(f"Error converting ClassSection to dictionary: {e}")
            # You might want to return None or raise the exception again after logging
            # raise
            return None  # Or return an empty dict {}

            
    def copy(self):
        return ClassSection(
            section=self.section,
            title=self.title,
            minCredit=self.minCredit,
            secCap=self.secCap,
            room=self.room,
            bldg=self.bldg,
            week_days=self.week_days,
            csm_start=self.csm_start,
            csm_end=self.csm_end,
            faculty1=self.faculty1,
            holdValue=self.holdValue,
            restrictions=self.avoid_classes.copy() if self.avoid_classes else None,
            assigned_meeting_time_indices=self.assigned_meeting_time_indices.copy() if self.assigned_meeting_time_indices else None
        )




# Function to create ClassSection objects from data
def create_class_sections_from_data(class_sections_data):
    class_sections = []
    seen_sec_names = set()  # Initialize a set to keep track of section names
    
    for section_data in class_sections_data:
        # Extract data from 'section_data' and create a ClassSection object
        sec_name = section_data.get('section', '')  # Updated to match the new column name
        if sec_name in seen_sec_names:
            # If the section name is already in the set, skip this iteration
            continue
        seen_sec_names.add(sec_name) # Add the section name to the set
        title = section_data.get('title', '')
        minCredit = section_data.get('minCredit', '')  # Updated to match the new column name
        secCap = section_data.get('secCap', '')  # Updated to match the new column name
        room = section_data.get('room', '')
        bldg = section_data.get('bldg', '')
        week_days = section_data.get('weekDays', '').strip() # Updated to match the new column name
        csm_start = section_data.get('csmStart', '')  # Updated to match the new column name
        csm_end = section_data.get('csmEnd', '')  # Updated to match the new column name
        faculty1 = section_data.get('faculty1', '')  # Updated to match the new column name
        holdValue = section_data.get('hold', '')  # Updated to match the new column name
        restrictions = section_data.get('restrictions', '')
        blocked_time_slots = section_data.get('blockedTimeSlots', '')  # Updated to match the new column name
        class_section = ClassSection(sec_name, title, minCredit, secCap, room, bldg, week_days, csm_start, csm_end, faculty1, holdValue ,restrictions,blocked_time_slots)
        class_sections.append(class_section)

    return class_sections






def update_class_sections_with_schedule(class_sections, class_timeslots, meeting_times):
    """
    Update the class sections with their scheduled day and time based on the optimization results.

    Args:
        class_sections (list): List of ClassSection objects.
        class_timeslots (dict): Dictionary of LpVariable binary variables representing class timeslots.
        meeting_times (MeetingTimes): Object containing meeting time data.
    """
    for class_section in class_sections:
        for day in ["M W F", "T Th"]:
            for start_time in meeting_times.choose_time_blocks(class_section.days, class_section.credits):
                variable = class_timeslots[class_section.section, day, start_time]
                if variable.varValue == 1:
                    # This class_section is scheduled for the specified day and time slot
                    class_section.scheduled_day = day
                    class_section.scheduled_time = start_time

# Create a list of ClassSection objects from the CSV data
def read_csv_and_create_class_sections(csv_filename):
    class_sections = []
    try:
        df = pd.read_csv(csv_filename)  # Read the CSV file with headers

        for index, row in df.iterrows():
            class_section = ClassSection(
                row['Term'], row['Section'], row['Title'], row['Location'], row['Meeting Info'],
                row['Faculty'], row['Available/Capacity'], row['Status'], row['Credits'],
                row['Academic Level'], row['Restrictions'], row['Blocked Time Slots'],
            )
            class_sections.append(class_section)

    except Exception as e:
        # Handle any exceptions that may occur during parsing
        print(f"Error processing CSV data: {str(e)}")

    return class_sections


def group_and_update_schedule(schedule_info_list):
    updated_schedules_list = []

    for schedule_info in schedule_info_list:
        # Extract the individual schedule from the current schedule_info
        individual_schedule = schedule_info['schedule']

        # Divide each schedule into groups of 3-credit and remaining classes
        three_credit_classes, remaining_classes = divide_schedules_by_credit(individual_schedule)

        # Group 3-credit classes based on common time patterns
        grouped_three_credit_classes = []
        for cls in three_credit_classes:
            # Since the cls dictionary already has the required keys, we can append it directly
            grouped_three_credit_classes.append(cls)

        # Directly append remaining classes, including 1-credit classes, to the schedule
        updated_schedule = grouped_three_credit_classes + remaining_classes

        # Update the schedule in the current schedule_info dictionary
        updated_schedules_list.append({
            'schedule': updated_schedule,
            'score': schedule_info['score'],
            'student_conflicts': schedule_info['student_conflicts'],  # include details if present
            'algorithm': schedule_info['algorithm'],
            'slot_differences': schedule_info['slot_differences'],
            'calendar_events': schedule_info.get('calendar_events', [])  # include calendar_events if present
        })

    return updated_schedules_list





def get_most_common_start_times(sections):
    # Separate MWF and TuTh timeslots
    mwf_start_times = [cls['timeslot'].split(' - ')[1] for cls in sections if 'M' in cls['timeslot'] or 'W' in cls['timeslot'] or 'F' in cls['timeslot']]
    tuth_start_times = [cls['timeslot'].split(' - ')[1] for cls in sections if 'Tu' in cls['timeslot'] or 'Th' in cls['timeslot']]

    # Find the most common start times
    most_common_mwf_time = Counter(mwf_start_times).most_common(1)[0][0] if mwf_start_times else None
    most_common_tuth_time = Counter(tuth_start_times).most_common(1)[0][0] if tuth_start_times else None

    return most_common_mwf_time, most_common_tuth_time



# Function to get all datetime slots for a class based on its week days and start time
def get_class_datetime_slots(week_days, start_time_str):
    start_time = datetime.strptime(start_time_str, '%I:%M%p').time()
    weekdays_map = {'M': 0, 'Tu': 1, 'W': 2, 'Th': 3, 'F': 4}
    datetime_slots = []
    for day in week_days.split():
        weekday = weekdays_map[day]
        class_datetime = datetime.combine(datetime.today(), start_time)
        days_ahead = (weekday - class_datetime.weekday()) % 7
        class_datetime += timedelta(days=days_ahead)
        datetime_slots.append(class_datetime)
    return datetime_slots

# Find the closest time slot for the 1-credit class
def find_closest_slot(cls_1_slots, cls_3_slots):
    min_diff = float('inf')
    for cls_1_slot in cls_1_slots:
        for cls_3_slot in cls_3_slots:
            diff = abs((cls_1_slot - cls_3_slot).total_seconds() / 3600)  # Difference in hours
            min_diff = min(min_diff, diff)
    return min_diff


def calculate_proximity_penalty(x, class_sections, available_timeslots):
    proximity_penalty = 0

    # Iterate over class sections
    for cls in class_sections:
        if cls['minCredit'] == '1' and cls['section'].endswith("_one_credit"):
            base_sec_name = cls['section'][:-11]  # Remove "_one_credit"

            # Find the corresponding 3-credit class
            corresponding_3_credit_cls = next((c for c in class_sections if c['section'] == base_sec_name), None)
            if corresponding_3_credit_cls:
                # Get timeslots for both 1-credit and 3-credit classes
                cls_1_slots = get_class_datetime_slots(cls['week_days'], cls['csm_start'])
                cls_3_slots = get_class_datetime_slots(corresponding_3_credit_cls['week_days'], corresponding_3_credit_cls['csm_start'])

                # Calculate time difference and apply penalty
                time_diff = find_closest_slot(cls_1_slots, cls_3_slots)
                for tsl in available_timeslots:
                    proximity_penalty += global_settings['move_penalty'] * time_diff * x[cls['section'], tsl]

    return proximity_penalty


def optimize_remaining_classes(class_sections, remaining_timeslots, used_timeslots):
    # Initial setup
    full_meeting_times = create_full_meeting_times()
    full_timeslots = [f"{mt['days']} - {mt['start_time']}" for mt in full_meeting_times]
    

   # Transform the used_timeslots to match the daily pattern
    transformed_used_timeslots = set()
    for used_tslot in used_timeslots:
        days, time = used_tslot.split(' - ')
        for day in days.split():
            transformed_used_timeslots.add(f"{day} - {time}")
            
            
    available_timeslots = [ts for ts in full_timeslots if ts not in transformed_used_timeslots]

    # Problem definition
    prob = LpProblem("Class_Scheduling_Remaining", LpMinimize)

    # Decision variables
    x = LpVariable.dicts("x", ((cls['section'], tsl) for cls in class_sections for tsl in available_timeslots), cat='Binary')

    # Objective function weights
    weights = {
        'schedule_penalty': 1.0,
        'blocked_slot_penalty': 1.0,
        'proximity_penalty': 1.0
    }

    # Define objective functions
    schedule_penalty = lpSum(x[cls['section'], tsl] for cls in class_sections for tsl in available_timeslots)
    
    # Check if student class mappings exist
    student_class_mapping = global_settings.get('student_class_mapping', {})

    if student_class_mapping:
        # Use student schedule conflict penalty if mappings exist
        blocked_slot_penalty = calculate_student_schedule_conflict_penalty(x, class_sections, available_timeslots, student_class_mapping,prob)
    else:
        # Use default blocked slot penalty if mappings do not exist
        blocked_slot_penalty = calculate_blocked_slot_penalty(x, class_sections, available_timeslots)
    
    proximity_penalty = calculate_proximity_penalty(x, class_sections, available_timeslots)

    # Adding weighted objectives to the problem
    prob += (weights['schedule_penalty'] * schedule_penalty +
             weights['blocked_slot_penalty'] * blocked_slot_penalty +
             weights['proximity_penalty'] * proximity_penalty)

    
    instructors = set()
    for cls in class_sections:
        instructors.add(cls['faculty1'])
        
    rooms = set()
    for cls in class_sections:
        rooms.add(cls['room'])

    # OneClassOneSlot Constraint
    for cls in class_sections:
        unique_constraint_name = f"OneClassOneSlotConstraint_{cls['section']}"
        prob += lpSum(x[(cls['section'], tsl)] for tsl in available_timeslots) == 1, unique_constraint_name

    # OneClassPerRoomPerSlot Constraint:
    for room in rooms:
        for tsl in available_timeslots:
            room_constraint_name = f"OneClassPerRoomPerSlot_{room}_{tsl}"
            prob += lpSum(x[(cls['section'], tsl)] for cls in class_sections if cls['room'] == room) <= 1, room_constraint_name

    # OneClassPerInstructorPerSlot Constraint:
    for instructor in instructors:
        for tsl in available_timeslots:
            instructor_constraint_name = f"OneClassPerInstructorPerSlot_{instructor}_{tsl}"
            prob += lpSum(x[(cls['section'], tsl)] for cls in class_sections if cls['faculty1'] == instructor) <= 1, instructor_constraint_name

            
    prob.solve()

    # Create a list to store the scheduled class sections
    scheduled_sections = []
    for cls in class_sections:
        for tsl in available_timeslots:
            if pulp.value(x[cls['section'], tsl]) == 1:
                scheduled_section = {
                    'section': cls['section'],
                    'timeslot': tsl,
                    'faculty1': cls['faculty1'],
                    'room': cls['room'],
                    'bldg': cls['bldg'],
                    'secCap' : cls['secCap'],
                }
                scheduled_sections.append(scheduled_section)

    # Create a dictionary to store optimization results
    optimization_results = {
        'message': 'Optimization for remaining classes complete',
        'scheduled_sections': scheduled_sections,
        'status': 'Success' if prob.status == LpStatusOptimal else 'Failure'
    }

    return optimization_results

# Supporting functions like calculate_blocked_slot_penalty, calculate_proximity_penalty, add_constraints, and process_scheduling_results should be defined separately.




def calculate_class_overlap(x, class_sections, global_settings, timeslots):
    class_overlap_penalty = 0
    constraint_counter = 0

    for cls in class_sections:
        if 'restrictions' in cls and cls['restrictions']:
            for tsl in timeslots:
                for other_cls_name in cls['restrictions']:
                    if (other_cls_name, tsl) in x:
                        constraint_counter += 1
                        # Increment penalty for each violation
                        class_overlap_penalty += global_settings['class_penalty'] * (x[cls['section'], tsl] + x[other_cls_name, tsl])

    return class_overlap_penalty



def calculate_timeslot_avoidance(x, class_sections, global_settings,timeslots):
    """  penalize for classes in unwanted timeslots """
    avoidance_penalty = 0

    # Iterate through each class section and its respective timeslots
    for cls in class_sections:
        for tsl in timeslots:
            # Check if the timeslot is one of the unwanted timeslots for the class
            if tsl in cls.get('unwanted_timeslots', []):
                # Increase the penalty if the class is scheduled in an unwanted timeslot
                avoidance_penalty += global_settings['timeslot_avoidance_penalty'] * x[cls['section'], tsl]

    return avoidance_penalty

def calculate_move_penalty(x, class_sections, meeting_times, timeslots):
    move_penalty = 0

    # Convert meeting_times to a lookup table for index
    # Make sure 'timeslot' is a key in each dictionary in meeting_times
    timeslot_indices = {mt['timeslot']: idx for idx, mt in enumerate(meeting_times) if 'timeslot' in mt}

    # Iterate through each class section
    for cls in class_sections:
        if cls.get('holdValue', 0) == 1:
            preferred_indices = cls.get('assigned_meeting_time_indices', [])
            for tsl in timeslots:
                if tsl in timeslot_indices and timeslot_indices[tsl] not in preferred_indices:
                    move_penalty += global_settings['move_penalty'] * x[cls['section'], tsl]

    return move_penalty




def calculate_blocked_slot_penalty(x, class_sections, timeslots):
    blocked_slot_penalty = 0

    # Iterate through each class section
    for cls in class_sections:
        unwanted_slots = cls.get('unwanted_timeslots', [])
        for tsl in timeslots:
            # If the timeslot is in the unwanted timeslots list for the class section
            if tsl in unwanted_slots:
                # Apply the penalty
                blocked_slot_penalty += global_settings['blocked_slot_penalty'] * x[cls['section'], tsl]

    return blocked_slot_penalty

def calculate_linked_sections_penalty(x, class_sections, timeslots):
    linked_sections_penalty = 0

    # Assuming 'linked_id' represents a common identifier for linked sections
    for cls in class_sections:
        if 'linked_id' in cls:
            linked_section = next((c for c in class_sections if c != cls and c.get('linked_id') == cls['linked_id']), None)
            if linked_section:
                for tsl in timeslots:
                    # Assuming global_settings['move_penalty'] is a globally accessible variable
                    index_diff = abs(timeslots.index(tsl) - timeslots.index(cls.get('preferred_timeslot', tsl)))
                    penalty_for_pair = index_diff * global_settings['move_penalty']
                    linked_sections_penalty += penalty_for_pair * (x[cls['section'], tsl] + x[linked_section['section'], tsl])

    return linked_sections_penalty




def calculate_student_schedule_conflict_penalty(x, class_sections, timeslots, student_class_mapping, prob):
    conflict_penalty = 0
    conflict_vars = {}

    # Iterate over each student and their classes
    for student, classes in student_class_mapping.items():
        # Iterate over all pairs of classes for each student
        for i, class1 in enumerate(classes):
            for j, class2 in enumerate(classes):
                if i < j and class1 in x and class2 in x:  # Ensure we only compare each pair once and classes are in x
                    for timeslot in timeslots:
                        # Create a new variable for the conflict
                        var_name = f'conflict_{student}_{class1}_{class2}_{timeslot}'
                        conflict_vars[var_name] = pulp.LpVariable(var_name, cat='Binary')

                        # Add constraints to ensure the conflict variable is 1 if both classes are scheduled at the same time
                        prob += conflict_vars[var_name] >= x[class1, timeslot] + x[class2, timeslot] - 1
                        prob += conflict_vars[var_name] <= x[class1, timeslot]
                        prob += conflict_vars[var_name] <= x[class2, timeslot]

                        # Add the conflict variable to the objective
                        conflict_penalty += conflict_vars[var_name]

    return conflict_penalty






def optimize_schedule(class_sections):
    
    # Helper function to get attribute or key value
    def get_value(item, key, default=None):
        return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)
    # Initial setup
    meeting_times = create_meeting_times()  # Define this function as needed
    rooms = set(cls['room'] for cls in class_sections if cls['room'].strip())
    instructors = set(cls['faculty1'] for cls in class_sections)
    timeslots = [f"{mt['days']} - {mt['start_time']}" for mt in meeting_times]
    
    # Problem definition
    prob = LpProblem("Class_Scheduling", LpMinimize)

    # Decision variables
    x = pulp.LpVariable.dicts("x", ((cls['section'], tsl) for cls in class_sections for tsl in timeslots), cat='Binary')

    # Objective function weights
    weights = {
        'class_overlap': 1.0,
        'timeslot_avoidance': 1.0,
        'move_penalty': 1.0,
        'blocked_slot_penalty': 1.0,
        'student_schedule_conflict': 1.0,  # Renamed for clarity
    }

    # Define objective functions and add them to the problem
    class_overlap_penalty = calculate_class_overlap(x, class_sections, meeting_times, timeslots)
    timeslot_avoidance_penalty = calculate_timeslot_avoidance(x, class_sections, meeting_times, timeslots)
    move_penalty = calculate_move_penalty(x, class_sections, meeting_times, timeslots)
    blocked_slot_penalty = calculate_blocked_slot_penalty(x, class_sections, meeting_times)
    
    # Check if student class mappings exist
    student_class_mapping = global_settings.get('student_class_mapping', {})
    # Only calculate student_schedule_conflict_penalty if student_class_mapping is present and non-empty
    student_schedule_conflict_penalty = 0
    if student_class_mapping:
        student_schedule_conflict_penalty = calculate_student_schedule_conflict_penalty(x, class_sections, timeslots, student_class_mapping, prob)

    # Objective function
    prob += (weights['class_overlap'] * class_overlap_penalty +
             weights['timeslot_avoidance'] * timeslot_avoidance_penalty +
             weights['move_penalty'] * move_penalty +
             weights['blocked_slot_penalty'] * blocked_slot_penalty +
             weights['student_schedule_conflict'] * student_schedule_conflict_penalty)


    # Constraints
    # Constraint: Each class must take exactly one timeslot
    for cls in class_sections:
        unique_constraint_name = f"OneClassOneSlotConstraint_{cls['section']}"
        prob += lpSum(x[(cls['section'], tsl)] for tsl in timeslots) == 1, unique_constraint_name

    # Constraint: No two classes can be in the same room at the same timeslot
    for room in rooms:
        for tsl in timeslots:
            room_constraint_name = f"OneClassPerRoomPerSlot_{room}_{tsl}"
            prob += lpSum(x[(cls['section'], tsl)] for cls in class_sections if cls['room'] == room) <= 1, room_constraint_name

    # Constraint: An instructor can only teach one class per timeslot
    for instructor in instructors:
        for tsl in timeslots:
            instructor_constraint_name = f"OneClassPerInstructorPerSlot_{instructor}_{tsl}"
            prob += lpSum(x[(cls['section'], tsl)] for cls in class_sections if cls['faculty1'] == instructor) <= 1, instructor_constraint_name


    # Solving the problem
    prob.solve()
    
    for variable in prob.variables():
        print(f"{variable.name} = {variable.varValue}")
        
    objective_value = value(prob.objective)

    # Results processing
    scheduled_sections = []
    for section, time in x:
        if value(x[section, time]) == 1:
            cls = next(cls for cls in class_sections if cls['section'] == section)
            scheduled_sections.append({
                'section': section,
                'timeslot': time,
                'faculty1': get_value(cls, 'faculty1'),
                'room': get_value(cls, 'room'),
                'secCap': get_value(cls, 'secCap', 'Unknown'),
                'bldg': get_value(cls, 'bldg', 'Unknown')
            })


    # Return results
    optimization_results = {
        'message': 'Optimization complete',
        'scheduled_sections': scheduled_sections,
        'status': LpStatus[prob.status]
    }
    return optimization_results




def get_weekday_date(reference_date, target_weekday):
    """
    Get the date for the target weekday based on the reference date (which is a Wednesday).
    """
    reference_weekday = reference_date.weekday()  # Monday is 0, Sunday is 6
    days_difference = target_weekday - reference_weekday
    return reference_date + timedelta(days=days_difference)

def annotate_failure_reasons(schedule, failure_report):
    # Check if failure_report[1] has values
    if  failure_report and  len(failure_report)>0:
        failure_details = {section_info[0]: section_info[1] for section_info in failure_report}
        for section in schedule:
            if section['section'] in failure_details:
                section['failure_reason'] = failure_details[section['section']]
            else:
                section['failure_reason'] = None  # No failure for this section
    else:
        # If failure_report[1] is empty, there are no failures
        for section in schedule:
            section['failure_reason'] = None  # No failure for this section

    return schedule

from ics import Calendar, Event
from datetime import datetime, timedelta

def process_calendar_data_to_ics(expanded_schedule):
    cal = Calendar()

    reference_date = datetime.today()
    while reference_date.weekday() != 2:  # Adjust to the nearest Wednesday
        reference_date += timedelta(days=1)

    weekday_map = {'M': 0, 'Tu': 1, 'W': 2, 'Th': 3, 'F': 4}

    for result in expanded_schedule:
        section_name = result['section']
        timeslot = result['timeslot']
        day, start_time = timeslot.split(' - ')

        # Check if day is 'nan' or not in weekday_map
        if day == 'nan' or day not in weekday_map:
            print(f"Skipping invalid day '{day}' in timeslot: {timeslot}")
            continue

        class_date = get_weekday_date(reference_date, weekday_map[day])
        start_datetime = datetime.combine(class_date, datetime.strptime(start_time, '%I:%M%p').time())
        duration = timedelta(hours=1)
        end_datetime = start_datetime + duration

        # Create an event
        event = Event()
        event.name = f"{section_name} Class"
        event.begin = start_datetime
        event.end = end_datetime
        event.location = result['room']
        event.description = f"Class with {result['faculty1']}"

        # Add event to calendar
        cal.events.add(event)

    # Convert the calendar to a string
    return str(cal)

def get_weekday_date(reference_date, weekday_offset):
    """
    Get the date for a specific weekday based on a reference date.
    """
    target_weekday = (reference_date.weekday() + weekday_offset) % 7
    return reference_date + timedelta(days=target_weekday - reference_date.weekday())

# Example usage:
# expanded_schedule = [...]  # Your expanded schedule data here
# ics_calendar_str = process_calendar_data_to_ics(expanded_schedule)
# with open("schedule.ics", "w") as file:
#     file.write(ics_calendar_str)



def process_calendar_data_expanded(expanded_schedule):
    calendar_data = []
    color_cache = {}

    reference_date = datetime.today()
    while reference_date.weekday() != 2:  # Adjust to the nearest Wednesday
        reference_date += timedelta(days=1)

    weekday_map = {'M': 0, 'Tu': 1, 'W': 2, 'Th': 3, 'F': 4}

    for result in expanded_schedule:
        section_name = result['section']
        timeslot = result['timeslot']
        day, start_time = timeslot.split(' - ')
        
        # Check if day is 'nan' or not in weekday_map
        if day == 'nan' or day not in weekday_map:
            print(f"Skipping invalid day '{day}' in timeslot: {timeslot}")
            continue

        class_date = get_weekday_date(reference_date, weekday_map[day])
        start_datetime = datetime.combine(class_date, datetime.strptime(start_time, '%I:%M%p').time())
        duration = timedelta(hours=1)
        end_datetime = start_datetime + duration

        course_prefix = section_name.split('-')[0]
        if course_prefix not in color_cache:
            color_cache[course_prefix] = str(string_to_color(course_prefix))
        color = color_cache[course_prefix]

        calendar_event = {
            'section_name': section_name,
            'start': start_datetime.strftime('%Y-%m-%dT%H:%M:%S'),
            'end': end_datetime.strftime('%Y-%m-%dT%H:%M:%S'),
            'faculty1': result['faculty1'],
            'room': result['room'],
            'color': color
        }
        calendar_data.append(calendar_event)

    return calendar_data


def process_calendar_data(three_credit_results, remaining_class_results):
    calendar_data = []
    color_cache = {}

    # Assume the current week's Monday as a reference date
    reference_date = datetime.today()
    while reference_date.weekday() != 0:  # Adjust to the nearest Monday
        reference_date -= timedelta(days=1)

    # Define a weekday map
    weekday_map = {'M': 0, 'Tu': 1, 'W': 2, 'Th': 3, 'F': 4}

    # Process classes
    for results in [three_credit_results['scheduled_sections'], remaining_class_results['scheduled_sections']]:
        for result in results:
            section_name = result['section']
            timeslot = result['timeslot']
            days, start_time = timeslot.split(' - ')
            start_time_obj = datetime.strptime(start_time, '%I:%M%p')

            for day in days.split():
                class_date = get_weekday_date(reference_date, weekday_map[day])
                start_datetime = datetime.combine(class_date, start_time_obj.time())
                # Determine the duration based on the days
                duration = timedelta(hours=1.5 if day in ['Tu', 'Th'] else 1)
                end_datetime = start_datetime + duration

                # Extract the course prefix and assign color
                course_prefix = section_name.split('-')[0]
                if course_prefix not in color_cache:
                    color_cache[course_prefix] = str(string_to_color(course_prefix))
                color = color_cache[course_prefix]

                calendar_event = {
                    'section_name': section_name,
                    'start': start_datetime.strftime('%Y-%m-%dT%H:%M:%S'),
                    'end': end_datetime.strftime('%Y-%m-%dT%H:%M:%S'),
                    'faculty1': result['faculty1'],
                    'room': result['room'],
                    'bldg': result['bldg'],
                    'color': color
                }
                if calendar_event and calendar_event['start'] and calendar_event['end'] and calendar_event['faculty1'] and calendar_event['room'] and calendar_event['bldg'] and calendar_event['color']:
                    calendar_data.append(calendar_event)

    return calendar_data

def get_weekday_date(reference_date, target_weekday):
    """
    Get the date for the target weekday based on the reference date (which is a Monday).
    """
    reference_weekday = reference_date.weekday()  # Monday is 0, Sunday is 6
    days_difference = target_weekday - reference_weekday
    return reference_date + timedelta(days=days_difference)

def combine_and_expand_schedule(three_credit_results, remaining_class_results, meeting_times, class_sections):
    combined_schedule = []

    # Expand and add three-credit class schedules
    for result in three_credit_results['scheduled_sections']:
        cls = next((c for c in class_sections if c.section== result['section']), None)
        if cls:
            # Extract days from the optimized timeslot
            optimized_days = result['timeslot'].split(' - ')[0].split()
            optimized_time = result['timeslot'].split(' - ')[1]
            # Expand each three-credit class into individual time slots based on its meeting days
            for day in optimized_days:
                day_specific_slot = f"{day} - {optimized_time}"
                combined_schedule.append({
                    'section': cls.section,
                    'timeslot': day_specific_slot,
                    'faculty1': cls.faculty1,
                    'room': cls.room,
                    'minCredit': cls.minCredit,
                    'unwanted_timeslots': cls.unwanted_timeslots,
                    'secCap': cls.secCap,
                    'bldg': cls.bldg,
                    'avoid_classes': cls.avoid_classes,
                    'hold_value': cls.holdValue
                })

    # Add one-credit class schedules with proper timeslot format
    # Continue to process each result in 'remaining_class_results'
    # Add class schedules with proper timeslot format
    for result in remaining_class_results['scheduled_sections']:
        section_name = result['section']

        # Determine the base section name (without '_one_credit' if present)
        base_section_name = section_name[:-11] if section_name.endswith('_one_credit') else section_name

        # Find the corresponding class section in 'class_sections'
        cls = next((c for c in class_sections if c.section == base_section_name), None)
        
        # Check if a corresponding class section was found
        if cls:
            # Append the formatted class section to 'combined_schedule'
            combined_schedule.append({
                'section': section_name,  # Use the original section name including '_one_credit' if applicable
                'timeslot': result['timeslot'],
                'faculty1': cls.faculty1,
                'room': cls.room,
                'minCredit': cls.minCredit,
                'unwanted_timeslots': cls.unwanted_timeslots,
                'hold_value': cls.holdValue,
                'secCap': cls.secCap,
                'bldg': cls.bldg,
                'avoid_classes': cls.avoid_classes,
                'hold_value': cls.holdValue
            })



        # 'combined_schedule' now contains the formatted schedule including the remaining class results

    return combined_schedule


def format_for_pulp(schedule):
    formatted_schedule = []

    for class_section in schedule:
        formatted_class = {
            'section': class_section['section'],
            'timeslot': class_section['timeslot'],
            'faculty1': class_section['faculty1'],
            'room': class_section.get('room', 'Unknown'),
            'secCap': class_section.get('secCap', 'Unknown'),
            'bldg': class_section.get('bldg', 'Unknown'),
            'avoid_classes': class_section.get('avoid_classes', []),
            'unwanted_timeslots': class_section.get('unwanted_timeslots', []),
            'holdValue': class_section.get('holdValue', 0)
        }

        # Additional attributes might be needed depending on your Pulp functions
        # For example, if minCredit or week_days are needed, add them here.

        formatted_schedule.append(formatted_class)

    return formatted_schedule


def expand_three_credit_class(class_result, meeting_times, class_section):
    # Logic to expand a three-credit class into individual time slots
    expanded_slots = []
    days = class_result['timeslot'].split(' ')[0].split()
    for day in days:
        expanded_slot = {
            'section_name': class_result['section_name'],
            'timeslot': f"{day} - {class_result['timeslot'].split(' ')[1]}",
            'faculty1': class_section.faculty1,
            'room': class_section.room,
            'minCredit': class_section.minCredit,
            'unwanted_timeslots': class_section.unwanted_timeslots
        }
        expanded_slots.append(expanded_slot)
    return expanded_slots

import random

# Define a cleaning function
def clean_text(text):
    # Remove BOM and other non-printable characters
    text = text.replace('\ufeff', '')  # BOM for UTF-8
    # Example: Remove all non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Strip leading/trailing whitespace and other specific cleaning as needed
    return text.strip()


def merge_schedules(schedule_1, schedule_2):
    merged_schedule = schedule_1.copy()  # Start with a copy of the first schedule

    # Iterate over the second schedule and add unique sections
    for section_2 in schedule_2:
        if section_2 not in merged_schedule:
            # Add the section from schedule_2 if it's not already in merged_schedule
            merged_schedule.append(section_2)

    return merged_schedule



def extract_schedule_from_pulp_results(pulp_results):
    extracted_schedule = []

    # Check if optimization was successful
    if pulp_results.get('status') == 'Success':
        scheduled_sections = pulp_results.get('scheduled_sections', [])

        for scheduled_section in scheduled_sections:
            # Extract and format each scheduled section
            formatted_section = {
                'section_name': scheduled_section.get('section_name'),
                'timeslot': scheduled_section.get('timeslot'),
                'faculty1': scheduled_section.get('faculty1'),
                'room': scheduled_section.get('room'),
                'sec_cap': scheduled_section.get('sec_cap', 'Unknown'),
                'bldg': scheduled_section.get('bldg', 'Unknown'),
                # Include any other relevant attributes you need from the Pulp results
            }
            extracted_schedule.append(formatted_section)

    return extracted_schedule



def create_timeslot_availability(individual):
    
    full_meeting_times = create_full_meeting_times()
    # Initialize dictionaries for availability
    timeslot_availability = {
        'M': {'MWF': [], 'other': []},
        'W': {'MWF': [], 'other': []},
        'F': {'MWF': [], 'other': []},
        'Tu': {'TuTh': [], 'other': []},
        'Th': {'TuTh': [], 'other': []}
    }
    room_availability = {}
    instructor_availability = {}

    # Populate timeslot availability based on full_meeting_times
    for timeslot in full_meeting_times:
        day = timeslot['days']
        time = f"{timeslot['start_time']} - {timeslot['end_time']}"
        if 'M' in day and 'W' in day and 'F' in day:
            timeslot_availability['M']['MWF'].append(time)
            timeslot_availability['W']['MWF'].append(time)
            timeslot_availability['F']['MWF'].append(time)
        elif 'Tu' in day and 'Th' in day:
            timeslot_availability['Tu']['TuTh'].append(time)
            timeslot_availability['Th']['TuTh'].append(time)
        else:
            if 'M' in day: timeslot_availability['M']['other'].append(time)
            if 'W' in day: timeslot_availability['W']['other'].append(time)
            if 'F' in day: timeslot_availability['F']['other'].append(time)
            if 'Tu' in day: timeslot_availability['Tu']['other'].append(time)
            if 'Th' in day: timeslot_availability['Th']['other'].append(time)

    # Populate room and instructor availability based on current individual schedule
    for cls in individual:
        timeslot = cls['timeslot']
        room_availability.setdefault(timeslot, set()).add(cls['room'])
        instructor_availability.setdefault(timeslot, set()).add(cls['faculty1'])

    return timeslot_availability, room_availability, instructor_availability

       

def extract_schedule_from_pulp_results(pulp_results):
    extracted_schedule = []

    # Check if optimization was successful
    if pulp_results.get('status') == 'Success':
        scheduled_sections = pulp_results.get('scheduled_sections', [])

        for scheduled_section in scheduled_sections:
            # Extract and format each scheduled section
            formatted_section = {
                'section_name': scheduled_section.get('section_name'),
                'timeslot': scheduled_section.get('timeslot'),
                'faculty1': scheduled_section.get('faculty1'),
                'room': scheduled_section.get('room'),
                'sec_cap': scheduled_section.get('sec_cap', 'Unknown'),
                'bldg': scheduled_section.get('bldg', 'Unknown'),
                # Include any other relevant attributes you need from the Pulp results
            }
            extracted_schedule.append(formatted_section)

    return extracted_schedule



def merge_class_details(schedule_result, class_section, day=None):
    # Merge details from class_section into the schedule result
    merged_result = schedule_result.copy()
    merged_result['faculty1'] = class_section.faculty1
    merged_result['room'] = class_section.room
    merged_result['minCredit'] = class_section.minCredit
    merged_result['unwanted_timeslots'] = class_section.unwanted_timeslots

    if day:
        # Adjust the timeslot for the specific day if provided
        merged_result['timeslot'] = f"{day} - {schedule_result['timeslot'].split(' ')[1]}"
    
    return merged_result




def determine_class_pattern(course_timeslots):
    # Count occurrences of each day in the timeslots
    day_count = {'M': 0, 'W': 0, 'F': 0, 'Tu': 0, 'Th': 0}
    for ts in course_timeslots:
        day = ts['timeslot'].split(' - ')[0]
        if day in day_count:
            day_count[day] += 1

    # Determine the pattern based on the counts
    if day_count['M'] > 0 and day_count['W'] > 0 and day_count['F'] > 0:
        return 'MWF'
    elif day_count['Tu'] > 0 and day_count['Th'] > 0:
        return 'TuTh'
    else:
        return 'Other'

def group_by_pattern(individual):
    # Group classes by their section and determine the overall pattern
    grouped_classes = {}
    for cls in individual:
        course_identifier = cls['section'].split('_')[0]
        grouped_classes.setdefault(course_identifier, []).append(cls)

    # Separate classes into MWF, TuTh, and Others based on their determined pattern
    mwf_classes = []
    tuth_classes = []
    other_classes = []

    for course_identifier, course_timeslots in grouped_classes.items():
        pattern = determine_class_pattern(course_timeslots)
        if pattern == 'MWF':
            mwf_classes.extend(course_timeslots)
        elif pattern == 'TuTh':
            tuth_classes.extend(course_timeslots)
        else:
            other_classes.extend(course_timeslots)

    return mwf_classes, tuth_classes, other_classes




def is_valid_individual(individual):
    full_meeting_times = create_full_meeting_times()
    room_assignments = {}
    instructor_assignments = {}
    failure_sections = []
    section_day_assignments = {}
    valid_timeslots = {f"{ts['days']} - {ts['start_time']}" for ts in full_meeting_times}

    # Group classes by their base section name for pattern adherence
    section_groupings = {}
    for cls in individual:
        base_section_name = cls['section'].split('_')[0]  # Use base name for pattern adherence
        section_groupings.setdefault(base_section_name, []).append(cls)

    for cls in individual:
        # Check if the class timeslot is valid
        if cls['timeslot'] not in valid_timeslots:
            failure_sections.append((cls['section'],"invalid timeslot"),cls)  # Use full section name

        # Check for multiple timeslot assignments on the same day for each section
        day = cls['timeslot'].split(' - ')[0]
        if day in section_day_assignments.get(cls['section'], set()):  # Use full section name
            failure_sections.append((cls['section'], "multiple timeslots on same day",cls))
        section_day_assignments.setdefault(cls['section'], set()).add(day)
        full_room = cls.get('bldg','') + ' ' + cls['room']
        # Check for room and instructor conflicts
        if cls['timeslot'] in room_assignments:
            if full_room in room_assignments[cls['timeslot']]:
                 # Check for room conflicts
                failure_sections.append((cls['section'],"room conflict"))
            elif cls['faculty1'] in instructor_assignments[cls['timeslot']]:
                # Check for instructor conflicts
                failure_sections.append((cls['section'],"instructor conflict",cls))
        room_assignments.setdefault(cls['timeslot'], set()).add(full_room)
        instructor_assignments.setdefault(cls['timeslot'], set()).add(cls['faculty1'])

        # Group classes by their base section name for pattern adherence
        section_groupings = {}
        for cls in individual:
            base_section_name = cls['section'].split('_')[0]  # Use base name for pattern adherence
            section_groupings.setdefault(base_section_name, []).append(cls)

        # ... (rest of the checks for invalid timeslot, multiple timeslots, room and instructor conflicts)

        # Check for pattern adherence for groups of sections
        for base_section_name, sections in section_groupings.items():
            pattern_sections = [cls for cls in sections if not cls['section'].endswith('_one_credit')]
            if any(int(cls['minCredit']) >= 3 for cls in pattern_sections):
                days = set(cls['timeslot'].split(' - ')[0] for cls in pattern_sections)
                times = set(cls['timeslot'].split(' - ')[1] for cls in pattern_sections)

                is_mwf = days.issubset({'M', 'W', 'F'}) and len(times) == 1
                is_tuth = days.issubset({'Tu', 'Th'}) and len(times) == 1

                if not (is_mwf or is_tuth):
                    failure_sections.append((base_section_name, "pattern adherence", pattern_sections[0]))  # Include an example class for reference

        # Return a tuple with a boolean for overall validity and a list of sections that failed validation
        return failure_sections




def follows_pattern(timeslot, pattern):
    """Check if a timeslot follows a specific pattern (MWF or TuTh)."""
    days = timeslot.split(' - ')[0]
    if pattern == 'MWF':
        return all(day in days for day in ['M', 'W', 'F'])
    elif pattern == 'TuTh':
        return all(day in days for day in ['Tu', 'Th'])
    return False




def split_class_sections(class_sections):
    three_credit_sections = []
    remaining_class_sections = []
    #class_sections = global_settings['class_sections_data']

    for class_section in class_sections:
        if int(class_section['minCredit']) == 4:
            # Create a copy of the class_section dictionary
            three_credit_section = class_section.copy()
            three_credit_section['minCredit'] = '3'

            # Create a modified copy for the one credit section
            one_credit_section = class_section.copy()
            one_credit_section['minCredit'] = '1'
            one_credit_section['section'] += "_one_credit"

            three_credit_sections.append(three_credit_section)
            remaining_class_sections.append(one_credit_section)

        elif int(class_section['minCredit']) == 3:
            three_credit_sections.append(class_section.copy())

        else:
            remaining_class_sections.append(class_section.copy())

    return three_credit_sections, remaining_class_sections


def count_slot_differences(pulp_schedule, ga_schedule):
    differences = 0
    pulp_sections = {section['section']: section['timeslot'] for section in pulp_schedule}
    
    for section in ga_schedule:
        section_name = section['section']
        if pulp_sections.get(section_name) != section['timeslot']:
            differences += 1

    return differences

@app.route('/meeting_times', methods=['GET'])
def get_meeting_times():
    meeting_times = create_meeting_times()
    return jsonify(meeting_times), 200


@app.route('/get_ical', methods=['GET'])
def get_ical():
    key = request.args.get('key')
    file_path = f'optimization_results/{key}.json'
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'Invalid or missing key'}), 400

    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        schedule = data['sorted_schedule']

    ical_content = convert_schedule_to_ical(schedule)

    response = make_response(ical_content)
    response.headers["Content-Disposition"] = f"attachment; filename=schedule_{key}.ics"
    response.headers["Content-Type"] = "text/calendar"
    return response


def weekday_to_date(start_date, weekday):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    start_weekday = start_date_obj.weekday()
    target_weekday = weekdays.index(weekday)
    days_to_add = (target_weekday - start_weekday) % 7
    return (start_date_obj + timedelta(days=days_to_add)).date()

def convert_schedule_to_ical(schedules, start_date=None, end_date=None):
    cal = Calendar()
    schedule = schedules[0]['schedule']

    days_map = {'M': 'Monday', 'Tu': 'Tuesday', 'W': 'Wednesday', 'Th': 'Thursday', 'F': 'Friday'}
    day_durations = {'Tu': timedelta(hours=1, minutes=30), 'Th': timedelta(hours=1, minutes=30), 'M': timedelta(hours=1), 'W': timedelta(hours=1), 'F': timedelta(hours=1)}

    if not start_date:
        start_date = datetime.today().strftime("%Y-%m-%d")
    if not end_date:
        end_date = (datetime.today() + timedelta(weeks=16)).strftime("%Y-%m-%d")

    for entry in schedule:
        days, time_str = entry['timeslot'].split(' - ')
        start_time = datetime.strptime(time_str, '%I:%M%p').time()
        for day_code in days.split():
            day_name = days_map.get(day_code)
            if day_name:
                duration = day_durations.get(day_code, timedelta(hours=1))
                end_time = (datetime.combine(datetime.today(), start_time) + duration).time()
                first_event_date = weekday_to_date(start_date, day_name)

                event = Event()
                event.name = f"{entry['section']}"
                event.begin = datetime.combine(first_event_date, start_time)
                event.end = datetime.combine(first_event_date, end_time)
                event.location = entry['bldg'] + " " + entry['room']
                subject = entry['section'].split('-')[0]
                event.categories = subject
                event.classification = subject
                event.organizer = entry['faculty1']
                event.description = f"Instructor:{entry['faculty1']} Capacity: {entry['secCap']} with Min-credits: {entry['minCredit']}"

                event.recurring = True
                event.recurrences = f"FREQ=WEEKLY;UNTIL={end_date.replace('-', '')}"

                cal.events.add(event)

    return str(cal)


def evaluateSchedule(individual):
    # Define penalties
    overlap_penalty = 300
    instructor_conflict_penalty = 300
    room_conflict_penalty = 300
    proximity_penalty = 2
    unwanted_timeslot_penalty = 3
    pattern_violation_penalty = 300
    mutual_exclusion_penalty = 2
    same_day_penalty = 300

    # Initialize scoring variables
    total_score = 0
    detailed_scores = {}
    class_timings = {}
    class_days = {}
    # Check if student class mappings exist
    student_class_mapping = global_settings.get('student_class_mapping', {})

    # Function to check if the class follows the MWF or TuTh pattern
    def follows_desired_pattern(course_identifier, timeslots):
        mwf_count = sum(['M' in ts and 'W' in ts and 'F' in ts for ts in timeslots])
        tuth_count = sum(['Tu' in ts and 'Th' in ts for ts in timeslots])

        # For 3-credit classes, either 2 TuTh or 3 MWF slots are acceptable
        if course_identifier.endswith('_3'):
            return (mwf_count == 3 and len(timeslots) >= 3) or (tuth_count == 2 and len(timeslots) >= 2)
        else:
            return mwf_count == len(timeslots) or tuth_count == len(timeslots)

    # Evaluate each class meeting
    for i, meeting in enumerate(individual):
        class_score = 0
        overlap_violations = 0
        instructor_conflicts = 0
        room_conflicts = 0
        proximity_issues = 0
        unwanted_timeslot_violations = 0
        mutual_exclusion_violations = 0
        total_unwanted_timeslot_violations =0 

        # Store timing and days for classes 3+ credits
        if int(meeting['minCredit']) >= 3:
            course_identifier = meeting['section'].split('_')[0]
            class_timings.setdefault(course_identifier, set()).add(meeting['timeslot'])
            days = meeting['timeslot'].split(' - ')[0]
            class_days.setdefault(course_identifier, set()).update(days.split())

        
        # If student class mappings exist, check for schedule conflicts
        if student_class_mapping:
            for student, classes in student_class_mapping.items():
                if meeting['section'] in classes:
                    for other_class in classes:
                        if other_class != meeting['section'] and any(meet['section'] == other_class and meet['timeslot'] == meeting['timeslot'] for meet in individual):
                            unwanted_timeslot_violations += 1
            total_unwanted_timeslot_violations += unwanted_timeslot_violations

        # If student mappings do not exist, calculate mutual exclusion and unwanted timeslot violations
        else:
            # Mutual exclusion violations
            mutually_exclusive_sections = meeting.get('avoid_classes', [])
            for other_meeting in individual:
                if other_meeting['section'] in mutually_exclusive_sections and meeting['timeslot'] == other_meeting['timeslot']:
                    mutual_exclusion_violations += 1

            # Unwanted timeslot violations
            for unwanted_timeslot in meeting.get('unwanted_timeslots', []):
                unwanted_days, unwanted_time = unwanted_timeslot.split(' - ')
                if unwanted_days in meeting['timeslot'] and unwanted_time == meeting['timeslot'].split(' - ')[1]:
                    unwanted_timeslot_violations += 1
            
        
        # Check for overlaps, conflicts, and violations
        for j, other_meeting in enumerate(individual):
            if i != j:
                if meeting['timeslot'] == other_meeting['timeslot']:
                    overlap_violations += 1
                    if meeting['room'] == other_meeting['room']:
                        room_conflicts += 1
                    if meeting['faculty1'] == other_meeting['faculty1']:
                        instructor_conflicts += 1

                # Mutual exclusion violations
                mutually_exclusive_sections = meeting.get('avoid_classes', [])
                if other_meeting['section'] in mutually_exclusive_sections:
                    mutual_exclusion_violations += 1

        # Proximity issues for 1-credit classes
        if meeting['minCredit'] == '1' and meeting['section'].endswith("_one_credit"):
            base_sec_name = meeting['section'][:-11]
            three_credit_class_meetings = [s for s in individual if s['section'] == base_sec_name]
            for cls_3_meeting in three_credit_class_meetings:
                time_diff = abs((datetime.strptime(meeting['timeslot'].split(' - ')[1], '%I:%M%p') - 
                                 datetime.strptime(cls_3_meeting['timeslot'].split(' - ')[1], '%I:%M%p')).total_seconds()) / 3600
                if time_diff > 2:
                    proximity_issues += 1

        # Unwanted timeslot violations
        for unwanted_timeslot in meeting.get('unwanted_timeslots', []):
            unwanted_days, unwanted_time = unwanted_timeslot.split(' - ')
            if unwanted_days in meeting['timeslot'] and unwanted_time == meeting['timeslot'].split(' - ')[1]:
                unwanted_timeslot_violations += 1

        # Calculate score for this class meeting
        class_score = (overlap_violations * overlap_penalty +
                       instructor_conflicts * instructor_conflict_penalty +
                       room_conflicts * room_conflict_penalty +
                       proximity_issues * proximity_penalty +
                       unwanted_timeslot_violations * unwanted_timeslot_penalty +
                       mutual_exclusion_violations * mutual_exclusion_penalty)

        # Update total score and details
        total_score += class_score
        detailed_scores[meeting['section']] = {
            'score': class_score,
            'overlap_violations': overlap_violations,
            'instructor_conflicts': instructor_conflicts,
            'room_conflicts': room_conflicts,
            'proximity_issues': proximity_issues,
            'unwanted_timeslot_violations': unwanted_timeslot_violations,
            'mutual_exclusion_violations': mutual_exclusion_violations
        }

    # Check for pattern matching and same day violations
    for course, timeslots in class_timings.items():
        if not follows_desired_pattern(course, timeslots):
            pattern_violation = pattern_violation_penalty * len(timeslots)
            total_score += pattern_violation
            for section in detailed_scores:
                if section.startswith(course):
                    detailed_scores[section]['pattern_violation'] = pattern_violation
        
        # Same day violations
        if len(class_days[course]) < len(timeslots):
            same_day_violation = same_day_penalty * (len(timeslots) - len(class_days[course]))
            total_score += same_day_violation
            for section in detailed_scores:
                if section.startswith(course):
                    detailed_scores[section].setdefault('same_day_violation', 0)
                    detailed_scores[section]['same_day_violation'] += same_day_violation

    
    
    return {'total_score': total_score,
            'detailed_scores': detailed_scores,
            'total_unwanted_timeslot_violations': total_unwanted_timeslot_violations,
            'unwanted_timeslot_penalty': unwanted_timeslot_penalty
    }
            


def validate_csv_for_class_section(csv_data):
    required_columns = ['section', 'title', 'minCredit', 'secCap', 'room', 'bldg', 'weekDays', 'csmStart', 'csmEnd', 'faculty1']
    optional_columns = ['hold', 'restrictions', 'blockedTimeSlots', 'weekDays']

    # Check if the list is not empty and is a list of dictionaries
    if not csv_data or not isinstance(csv_data, list) or not all(isinstance(item, dict) for item in csv_data):
        return "CSV data is empty or not in the expected format (list of dictionaries).", False

    # Check if there's at least one row in the data
    if not csv_data:
        return "CSV data is empty.", False

    # Check the keys in the first row (dictionary) of csv_data
    first_row_keys = csv_data[0].keys()
    missing_columns = [col for col in required_columns if col not in first_row_keys]
    
    if missing_columns:
        return f"Missing required columns: {', '.join(missing_columns)}", False

    return "CSV data is valid for ClassSection.", True


def custom_mutate(individual, mutpb):
    section_timeslots_map = {}
    full_meeting_times = create_full_meeting_times()
    for class_section in individual:
        section_name = class_section['section']
        section_timeslots_map.setdefault(section_name, []).append(class_section)

    for i in range(len(individual)):
        class_section = individual[i]
        # Directly use the mutation probability provided (mutpb) without adjustment for failed_sections
        if random.random() < mutpb:
            minCredit = int(class_section['minCredit'])
            pattern_timeslots = []
            mwf_timeslots = []
            tuth_timeslots = []

            if minCredit >= 3:
                all_timeslots = section_timeslots_map[section_name]
                is_mwf = all(['M' in ts['timeslot'] and 'W' in ts['timeslot'] and 'F' in ts['timeslot'] for ts in all_timeslots])
                is_tuth = all(['Tu' in ts['timeslot'] and 'Th' in ts['timeslot'] for ts in all_timeslots])

                if random.random() < 0.5:  # Change timeslot within the same pattern
                    if is_mwf:
                        pattern_timeslots = [ts for ts in full_meeting_times if 'M' in ts['days'] and 'W' in ts['days'] and 'F' in ts['days']]
                    elif is_tuth:
                        pattern_timeslots = [ts for ts in full_meeting_times if 'Tu' in ts['days'] and 'Th' in ts['days']]

                    if pattern_timeslots:
                        new_timeslot = random.choice(pattern_timeslots)
                        for ts in all_timeslots:
                            ts['timeslot'] = f"{new_timeslot['days']} - {new_timeslot['start_time']}"
                else:  # Switch pattern
                    if is_mwf:
                        tuth_timeslots = [ts for ts in full_meeting_times if 'Tu' in ts['days'] and 'Th' in ts['days']]
                    elif is_tuth:
                        mwf_timeslots = [ts for ts in full_meeting_times if 'M' in ts['days'] and 'W' in ts['days'] and 'F' in ts['days']]

                    if (is_mwf and tuth_timeslots) or (is_tuth and mwf_timeslots):
                        new_timeslot = random.choice(tuth_timeslots if is_mwf else mwf_timeslots)
                        for ts in all_timeslots:
                            ts['timeslot'] = f"{new_timeslot['days']} - {new_timeslot['start_time']}"

            else:
                current_day = class_section['timeslot'].split(' - ')[0]
                same_day_timeslots = [ts for ts in full_meeting_times if current_day in ts['days']]
                if same_day_timeslots:
                    new_timeslot = random.choice(same_day_timeslots)
                    class_section['timeslot'] = f"{new_timeslot['days']} - {new_timeslot['start_time']}"

    return individual,
from datetime import datetime

def preprocess_input_data(class_sections):
    """
    Adjusts the input data for each class section to include a 'timeslot' key
    that combines 'week_days', 'csm_start', and 'csm_end'.
    
    Args:
        class_sections (list): List of dictionaries, each representing a class section.
    
    Returns:
        list: The adjusted list of class sections with added 'timeslot' keys.
    """
    adjusted_class_sections = []
    for section in class_sections:
        # Combine 'week_days', 'csm_start', and 'csm_end' into a 'timeslot' key
        week_days = section.get('week_days', '')
        csm_start = section.get('csm_start', '')
        csm_end = section.get('csm_end', '')
        
        # Only combine if all parts are present
        if week_days and csm_start and csm_end:
            timeslot = f"{week_days} - {csm_start} - {csm_end}"
        else:
            # Handle cases where the class might not have a defined timeslot
            timeslot = "Undefined"
        
        # Copy the original section dictionary to avoid mutating the original data
        adjusted_section = section.copy()
        adjusted_section['timeslot'] = timeslot
        
        # Add the adjusted section to the new list
        adjusted_class_sections.append(adjusted_section)
    
    return adjusted_class_sections





def custom_crossover(ind1, ind2):
    full_meeting_times = create_full_meeting_times()

    # Group classes by pattern
    mwf1, tuth1, other1 = group_by_pattern(ind1)
    mwf2, tuth2, other2 = group_by_pattern(ind2)

    # Randomly select crossover points for MWF and TuTh patterns
    crossover_point_mwf = random.randint(1, min(len(mwf1), len(mwf2)) - 1)
    crossover_point_tuth = random.randint(1, min(len(tuth1), len(tuth2)) - 1)

    # Swap MWF sections if it results in valid schedules
    mwf1_new = mwf1[:crossover_point_mwf] + mwf2[crossover_point_mwf:]
    mwf2_new = mwf2[:crossover_point_mwf] + mwf1[crossover_point_mwf:]
    if is_valid_individual(mwf1_new + tuth1 + other1)[0] and is_valid_individual(mwf2_new + tuth2 + other2)[0]:
        mwf1, mwf2 = mwf1_new, mwf2_new

    # Swap TuTh sections if it results in valid schedules
    tuth1_new = tuth1[:crossover_point_tuth] + tuth2[crossover_point_tuth:]
    tuth2_new = tuth2[:crossover_point_tuth] + tuth1[crossover_point_tuth:]
    if is_valid_individual(mwf1 + tuth1_new + other1)[0] and is_valid_individual(mwf2 + tuth2_new + other2)[0]:
        tuth1, tuth2 = tuth1_new, tuth2_new

    # Swap other classes if it results in valid schedules
    crossover_point_other = random.randint(1, len(other1))
    other1_new = other1[:crossover_point_other] + other2[crossover_point_other:]
    other2_new = other2[:crossover_point_other] + other1[crossover_point_other:]
    if is_valid_individual(mwf1 + tuth1 + other1_new)[0] and is_valid_individual(mwf2 + tuth2 + other2_new)[0]:
        other1, other2 = other1_new, other2_new

    # Reconstruct ind1 and ind2 from the modified groups
    ind1_new = mwf1 + tuth1 + other1
    ind2_new = mwf2 + tuth2 + other2

    # Invalidate the fitness of the modified individuals
    del ind1.fitness.values
    del ind2.fitness.values

    # Update the individuals only if the new combinations are valid
    ind1[:] = ind1_new
    ind2[:] = ind2_new

    return ind1, ind2



# Assuming create_individual, create_full_meeting_times, custom_crossover, evaluateSchedule, and divide_schedules_by_credit are defined elsewhere

def run_genetic_algorithm(combined_expanded_schedule, report, ngen=30, pop_size=50, cxpb=0.3, mutpb=0.2):
    try:
        # Create necessary data
        full_meeting_times_data = create_full_meeting_times()

        # Setup the DEAP toolbox
        toolbox = base.Toolbox()
        
        # Register individual and population creation methods
        toolbox.register("individual", create_individual, report, combined_expanded_schedule)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register custom mutate method without relying on failed sections
        toolbox.register("mutate", lambda ind: custom_mutate(ind, mutpb))

        # Register mate and select methods
        toolbox.register("mate", custom_crossover)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Adjust the evaluate function to use only the total score from evaluateSchedule's output
        def evaluate_wrapper(individual):
            evaluation_results = evaluateSchedule(individual)
            return evaluation_results['total_score'],

        toolbox.register("evaluate", evaluate_wrapper)

        # Create initial population
        population = toolbox.population(n=pop_size)

        # Evaluate the initial population's fitness
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Collecting statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run genetic algorithm
        final_population, logbook = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=stats, verbose=True)

                      
            
        # Process final population
        sorted_population = sorted(final_population, key=lambda ind: ind.fitness.values[0])  # Sort in ascending order

        # Identify top unique schedules
        top_unique_schedules = []
        used_scores = set()
        for ind in sorted_population:
            fitness_score = ind.fitness.values[0]
            if fitness_score not in used_scores:
                top_unique_schedules.append((ind, fitness_score))
                used_scores.add(fitness_score)
                if len(top_unique_schedules) == 5:
                    break

        # Take the top GA solution and split back into 3 and 1 credit classes and return them
        #three_credit_classes, remaining_classes = divide_schedules_by_credit(top_unique_schedules[0][0])

        return top_unique_schedules
    except Exception as e:
        # Handle the error gracefully and provide diagnostic information
        print(f"An error occurred during the genetic algorithm execution: {e}")
        initial_sorted_population = sorted(
            population, 
            key=lambda ind: ind.fitness.values if ind.fitness.valid else float('inf')
        )
        top_unique_schedules = []
        for ind in initial_sorted_population:
            if ind.fitness.valid:
                fitness_score = ind.fitness.values[0]
                top_unique_schedules.append((ind, fitness_score))
                break  # Only take the top individual, like the successful block
        return top_unique_schedules if top_unique_schedules else [("An error occurred, and no valid individuals are available.", float('inf'))]


@app.route('/get_csv', methods=['GET'])
def get_csv():
    key = request.args.get('key')
    file_path = f'optimization_results/{key}.json'
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'Invalid or missing key'}), 400

    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        schedule = data['sorted_schedule']

    csv_content = convert_schedule_to_csv(schedule)

    response = make_response(csv_content)
    response.headers["Content-Disposition"] = f"attachment; filename=schedule_{key}.csv"
    response.headers["Content-Type"] = "text/csv"
    return response

def convert_schedule_to_csv(schedule):
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['Section', 'Timeslot', 'Faculty', 'Room', 'Building'])  # Add other headers as needed
    for entry in schedule:
        cw.writerow([entry['section'], entry['timeslot'], entry['faculty1'], entry['room'], entry['bldg']])  # Match with your data structure
    return si.getvalue()


@app.route('/optimize', methods=['POST'])
def optimize():
    try:
    
        global global_settings
        data = request.get_json()
        
        unique_key = str(uuid.uuid4())
        
        global_settings['class_sections_data'] = data.get('classData', [])
        global_settings['class_penalty'] = data.get('classPenalty', 0)
        global_settings['move_penalty'] = data.get('movePenalty', 0)
        global_settings['blocked_slot_penalty'] = data.get('blockedSlotPenalty', 0)
        global_settings['hold_penalty'] = data.get('holdPenalty', 0)
        
        
        message, is_valid = validate_csv_for_class_section(global_settings['class_sections_data'])
        
        if not is_valid:
            return jsonify({'message':  message }), 400

        class_sections = create_class_sections_from_data(global_settings['class_sections_data'])
        
        # Convert class section instances to dictionaries
        class_sections_dictionaries = [class_section.to_dictionary() for class_section in class_sections]

        ## original score
        orig_score = evaluateSchedule(preprocess_input_data(class_sections_dictionaries))
       
        three_credit_sections, remaining_class_sections =  split_class_sections(class_sections_dictionaries)

        
        # Optimize the 3-credit classes with Pulp
        three_credit_results = optimize_schedule(three_credit_sections)

        # Create a set of used timeslots
        used_timeslots = set()
        for section in three_credit_results['scheduled_sections']:
            used_timeslots.add(section['timeslot'])
            
        # Calculate remaining timeslots
        all_timeslots = [f"{day} - {mt['start_time']}" for mt in create_full_meeting_times() for day in mt['days'].split()]
        remaining_timeslots = [ts for ts in all_timeslots if ts not in used_timeslots]

        # Optimize the remaining classes with Pulp
        remaining_class_results = optimize_remaining_classes(remaining_class_sections, remaining_timeslots, used_timeslots)
        calendar_events = process_calendar_data(three_credit_results, remaining_class_results)

        # Combine and expand the schedules
        combined_expanded_schedule = combine_and_expand_schedule(three_credit_results, remaining_class_results, create_full_meeting_times(), class_sections)

        #lets validate the results against he constraints using the GA method
    
        failure_report=is_valid_individual(combined_expanded_schedule)
        
        # use the genetic algorithm evalutaion function to evaluate the schedule
        pulp_evaluation_results = evaluateSchedule(combined_expanded_schedule)
        pulp_score = pulp_evaluation_results['total_score']
        student_conflicts = pulp_evaluation_results['detailed_scores'].get('student_conflicts', 0)
        
        
        #marked_combined_expanded_schedule
        #all_schedules = ({'schedule': combined_expanded_schedule, 'score': pulp_score, 'algorithm': 'PuLP', 'calendar_events':calendar_events , 'slot_differences': 0})
        
        #marked_combined_expanded_schedule
        
        
        
        all_schedules = [{
            'schedule': combined_expanded_schedule,
            'score': pulp_score,
            'student_conflicts': student_conflicts,
            'algorithm': 'PuLP',
            'calendar_events': calendar_events,
            'slot_differences': 0
        }]
            
        # Run the genetic algorithm to optimize the schedule further
        top_unique_schedules = run_genetic_algorithm(combined_expanded_schedule,failure_report)
        
        
         # Process each top GA schedule to add to all_schedules
        for ga_schedule in top_unique_schedules:
            # Split schedules by credit
            three_credit_classes, remaining_classes = divide_schedules_by_credit(ga_schedule[0])

            if  is_valid_individual(ga_schedule[0]):
                # Prepare the three-credit class results in the required format for processing calendar events
                three_credit_results_formatted = {
                    'message': 'Optimization complete',
                    'scheduled_sections': three_credit_classes,
                    'status': 'Optimal'
                }

                # Prepare the remaining class results in the required format for processing calendar events
                remaining_class_results_formatted = {
                    'message': 'Optimization for remaining classes complete',
                    'scheduled_sections': remaining_classes,
                    'status': 'Success'
                }
                
                # Generate calendar events for the GA schedule
                calendar_events_ga = process_calendar_data(three_credit_results_formatted, remaining_class_results_formatted)
                ga_res=evaluateSchedule(ga_schedule[0])
                
                # Add GA schedule to all_schedules
                all_schedules.append({
                    'schedule': ga_schedule[0],
                    'score': ga_schedule[1],
                    'algorithm': 'GA',
                    'student_conflicts':ga_res['total_unwanted_timeslot_violations'],
                    'calendar_events': calendar_events_ga,  # GA calendar events
                    'slot_differences': 0  # Assuming ga_schedule[2] holds slot differences
                })

            

        all_schedules_sorted = sorted(all_schedules, key=lambda x: x['score'])
        all_schedules_sorted = all_schedules
        final_schedule = group_and_update_schedule(all_schedules_sorted)

        # Combine PuLP and GA results for the response
        final_results = {
            'unique_key': unique_key,
            'sorted_schedule': final_schedule,  # This might be your optimized schedule
            'message': 'Optimization completed'
        }
        
        write_results_to_json(unique_key, final_results)
        return jsonify(final_schedule)

    except Exception as e:
            # Handle specific exceptions or general exceptions
            error_message = f"Error: {str(e)}"  # Format the message based on the exception type
            return jsonify({'message': error_message}), 400



def write_results_to_json(key, results):
    # Define a directory to store the JSON files
    directory = 'optimization_results'
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, f"{key}.json")
    with open(filepath, 'w') as json_file:
        json.dump(results, json_file, indent=4)


def divide_schedules_by_credit(schedule, credit_threshold=3):
    three_credit_classes = []
    remaining_classes = []
    section_groupings = {}

    for class_section in schedule:
        if class_section['section'].endswith('_one_credit'):
             remaining_classes.append(class_section)
             continue
        base_section_name = class_section['section'].split('_')[0]
        section_groupings.setdefault(base_section_name, []).append(class_section)

    for base_section_name, sections in section_groupings.items():
        if any(int(cls['minCredit']) >= credit_threshold for cls in sections):
            mwf_time, tuth_time = get_most_common_start_times(sections)
            if mwf_time:
                mwf_sections = [cls for cls in sections if 'M' in cls['timeslot'] or 'W' in cls['timeslot'] or 'F' in cls['timeslot']]
                if mwf_sections:
                    three_credit_classes.append({
                        'section': base_section_name,
                        'timeslot': 'M W F - ' + mwf_time,
                        'minCredit': mwf_sections[0]['minCredit'],
                        'faculty1': mwf_sections[0]['faculty1'],
                        'room': mwf_sections[0]['room'],
                        'bldg': mwf_sections[0]['bldg'],
                        'secCap': mwf_sections[0]['secCap']
                    })
            if tuth_time:
                tuth_sections = [cls for cls in sections if 'Tu' in cls['timeslot'] or 'Th' in cls['timeslot']]
                if tuth_sections:
                    three_credit_classes.append({
                        'section': base_section_name,
                        'timeslot': 'Tu Th - ' + tuth_time,
                        'minCredit': tuth_sections[0]['minCredit'],
                        'faculty1': mwf_sections[0]['faculty1'],
                        'room': mwf_sections[0]['room'],
                        'bldg': mwf_sections[0]['bldg'],
                        'secCap': mwf_sections[0]['secCap']
                    })
        else:
            remaining_classes.extend(sections)

    return three_credit_classes, remaining_classes


def process_uploaded_data(uploaded_file_data):
    try:
        # Create a DataFrame from the uploaded data
        df = pd.read_csv(StringIO(uploaded_file_data))

        # Define the columns to keep
        columns_to_keep = [
            'sec name','title', 'min Credit','sec Cap', 'room',
            'bldg', 'week Days', 'CSM start', 'CSM end','faculty1','Restrictions','Blocked Time Slots'
        ]

        # Check if all required columns exist in the DataFrame
        missing_columns = set(columns_to_keep) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Columns not found: {', '.join(missing_columns)}")

        # Select the desired columns
        filtered_df = df[columns_to_keep]
        # Replace NaN values with ""
        filtered_df = filtered_df.fillna("")

        # Convert the filtered DataFrame to a list of dictionaries
        class_sections_data = filtered_df.to_dict(orient='records')

        return class_sections_data

    except Exception as e:
        # Handle any exceptions that may occur during parsing
        print(f"Error processing CSV data: {str(e)}")
        return None




@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read the file content into a string
        file_content = file.read().decode('utf-8')
        
        # Process the file content as needed, and prepare your response
        response_data = process_uploaded_data(file_content)
        return jsonify(response_data), 200

    return jsonify({'error': 'Unknown error occurred'}), 500



@app.route('/load-schedule', methods=['GET'])
def load_schedule():
    # Read the CSV file and parse it into a list of dictionaries
    csv_filename = 'your_schedule.csv'  # Adjust the filename
    schedule_data = []

    with open(csv_filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            schedule_data.append(row)

    # Send the parsed data as JSON
    return jsonify(schedule_data)


@app.route('/', methods=['GET', 'POST'])
def index():
    return "Hello, World!"

@app.route('/load_original', methods=['POST'])
@cross_origin()  # This decorator enables CORS specifically for this route
def load_original():

    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read the uploaded file data
        uploaded_file_data = file.read().decode('utf-8')

        # Attempt to convert the uploaded file data from CSV to JSON
        try:
            csv_data = io.StringIO(uploaded_file_data)
            csv_reader = csv.DictReader(csv_data)
            data_rows = [row for row in csv_reader]
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        # Define the required columns
        required_columns = {'sec name', 'title', 'min Credit', 'sec Cap', 'room', 'bldg', 'week Days', 'CSM start', 'CSM end', 'faculty1', 'Restrictions', 'Blocked Time Slots'}
        existing_columns = set(csv_reader.fieldnames)

        # Determine which required columns are missing
        missing_columns = required_columns - existing_columns
        added_columns = False

        # Add missing columns with default values
        if missing_columns:
            added_columns = True
            for row in data_rows:
                for column in missing_columns:
                    row[column] = None  # Assign a default value or None

        # Convert the data back to CSV or JSON as required by your application
        # For simplicity, we're returning JSON
        response_data = {
            'data': data_rows,
            'message': 'Missing columns added.' if added_columns else 'All required columns present.'
        }

        return jsonify(response_data), 200

    return jsonify({'error': 'Unknown error occurred'}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
