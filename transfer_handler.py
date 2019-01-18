# coding: utf-8

import numpy as np
from collections import Counter

import time

from sklearn.preprocessing import normalize

#########################################################################################################
######################################## Transfer #######################################################
def calGapDates(before, after):
    gap = 0
    s = before
    while compareTo(s, after) <= 0:
        gap += 1
        s = addDay(s, 1)
    return gap
def calMonthDays(year, month):
    if month == 2:
        if year % 4 == 0:
            days = 29
        else:
            days = 28
    elif month == 4 or month == 6 or month == 9 or month == 11:
        days = 30
    else:
        days = 31
    return days
def addDay(date, add):
    year=int(date[:4])
    month=int(date[5:7])
    day=int(date[8:10])

    month_days = calMonthDays(year, month)
    day += add
    while day > month_days:
        if day > month_days:
            month += 1
            day -= month_days

        if month > 12:
            year += 1
            month -= 12

        month_days = calMonthDays(year, month)

    added_date='%04d-%02d-%02d'%(year,month,day)
    return added_date
def compareTo(from_date, to_date):
    if int(from_date[:4]) < int(to_date[:4]):
        return -1
    elif int(from_date[:4]) > int(to_date[:4]):
        return 1
    else:
        if int(from_date[5:7]) < int(to_date[5:7]):
            return -1
        elif int(from_date[5:7]) > int(to_date[5:7]):
            return 1
        else:
            if int(from_date[8:10]) < int(to_date[8:10]):
                return -1
            elif int(from_date[8:10]) > int(to_date[8:10]):
                return 1
            else:
                return 0

def reduceDay(date, days=1):
    year = int(date[:4])
    month = int(date[5:7])
    day = int(date[8:10])

    day -= days
    while day < 1:
        month -= 1
        if month < 1:
            year -= 1
            month = 12
            day += 31
        else:
            if month == 2:
                if year % 4 == 0:
                    day += 29
                else:
                    day += 28
            elif (month == 4 or month == 6 or month == 9 or month == 11):
                day += 30
            else:
                day += 31

    return '%04d-%02d-%02d' % (year, month, day)
def transferToSeconds(logtime):
    time = logtime[14:]

    min = int(time[:2])
    sec = int(time[3:])

    return min * 60 + sec

def slicingDataOnDays(users_in_days_data, dates, id_axis, el_axis, term):
    days_data = [[] for i in range(len(dates))]
    for user_data in users_in_days_data:
        user_id = user_data[id_axis-1]
        user_data = user_data[id_axis:]
        # el_size = int(len(user_data)/(len(dates)/term))
        for i in range(int(len(dates)/term)):
            days_data[i].append([user_id]+user_data[i*el_axis:i*el_axis+el_axis])

    # days_data=[[[id, elements]]]
    return days_data

# actions_in_period={id={elements}}
def periodActionsToArray(actions, t_elements):
    data = []
    for user_id in actions.keys():
        user = [user_id]
        user += [actions[user_id][element] for element in t_elements]
        data.append(user)
    return data
# actions_in_days={id={dates:{elements}}}
def daysActionsToArray(actions, dates, t_elements):
    data = []
    for user_id in actions.keys():
        user = [user_id]
        for date in dates:
            if date in actions[user_id].keys() :
                aday = actions[user_id][date]
                user += [aday[element] for element in t_elements]
            else:
                user += [0 for element in t_elements]
        data.append(user)
    return data

# actions_in_period={id={elements}}
def periodActionsToDict(actions, t_elements):
    data = {}
    for line in actions:
        el = {}
        i = 1
        for element in t_elements:
            el[element] = float(line[i])
            i+=1
        data[line[0]] = el
    return data
# actions_in_days={id={dates:{elements}}}
def daysActionsToDict(actions, dates, t_elements):
    if actions is None: return None
    data = {}
    for line in actions:
        date = {}
        for d_index, day in enumerate(range(0, len(line)-1, len(t_elements))):
            el = {}
            for i, element in enumerate(t_elements):
                el[element] = float(line[day+i+1])
            date[dates[d_index]] = el
        data[line[0]] = date
    return data
def transferUsersToTermsData(users, dates, t_elements, terms):
    d_elements = t_elements
    if 'churn' in t_elements: d_elements.remove('churn')
    if 'accessdays' in t_elements: d_elements.remove('accessdays')

    users_data = []
    sum_users_data = []
    vec_users_data = []

    churns = []
    uids = []
    if not terms == 1:
        for user_id in users.keys():
            d_data = []
            sum_data = []
            vec_data = []
            ds_data = []
            dv_data = []
            aday_access = 0
            for i, date in enumerate(dates):
                if date in users[user_id].keys():
                    d = [users[user_id][date][element] for element in d_elements]
                else:
                    d = [0 for element in d_elements]

                d_data.append(d)
                ds_data.append(d)
                if d[0] > 0: aday_access += 1
                dv_data += d

                if (i+1) % terms == 0 and not i == 0:
                    sum_data.append(np.sum(ds_data, axis=0).tolist() + [aday_access])
                    vec_data.append(dv_data)
                    ds_data = []
                    dv_data = []
                    aday_access = 0
            users_data.append(d_data)
            sum_users_data.append(sum_data)
            vec_users_data.append(vec_data)
            churns.append(users[user_id]['churn'])
            uids.append(user_id)
    elif terms == 1:
        for user_id in users.keys():
            d_data = []
            for i, date in enumerate(dates):
                if date in users[user_id].keys():
                    d = [users[user_id][date][element] for element in d_elements]
                else:
                    d = [0 for element in d_elements]
                d_data.append(d)
            users_data.append(d_data)
            churns.append(users[user_id]['churn'])
            uids.append(user_id)

    terms_dates = []
    if terms == 1:
        las_train_data = {'origin': users_data, 'dates': dates, 'terms_dates': dates, 'uids': uids, 'churns': churns}
    else:
        len_dates = len(dates) - 1
        for i in range(0, len_dates, terms):
            terms_dates.append(dates[i])

        las_train_data = {'origin': users_data,
                          'sum': sum_users_data,
                          'vec': vec_users_data,
                        'dates': dates, 'terms_dates': terms_dates, 'uids': uids, 'churns': churns}

    return las_train_data

def transferOnehotLabels(labels):
    labels = np.array(labels, dtype=np.int_)
    if -1 in labels: labels[np.where(labels == -1)] = max(labels)+1
    max_range = int(max(labels))+1
    onehot_labels=[]
    for label in labels:
        onehot_label = np.zeros((1, max_range))[0].tolist()
        onehot_label[int(label)] = 1
        onehot_labels.append(onehot_label)
    return onehot_labels

def checkDate(start_date, end_date, terms):
    if compareTo(start_date[:10], end_date[:10]) > 0: return False
    if calGapDates(start_date, end_date)%terms == 0: return True
    else : return False

def quantilize(scalar, quantilizer):
    quantilized = np.zeros((1, len(quantilizer))).tolist()[0]
    b = 0
    for i, q in enumerate(quantilizer):
        if b < scalar and scalar <= q:
            quantilized[i] = 1
            break
        b = q
    return quantilized
def transferToQuantilization(data, distribution=10):
    term = int(100/distribution)
    feature_size = len(data[0])

    data = np.array(data)
    quantilizer=[]
    for f in range(feature_size):
        f_data = np.unique(data[:,f])
        q=[]
        for i in range(distribution):
            q.append(np.percentile(f_data, term*(i+1)))
        quantilizer.append(q)

    trans_data = []
    for d in data:
        t_data = []
        for i, q in enumerate(quantilizer):
            t_data.append(quantilize(d[i], q))
        trans_data.append(t_data)
    trans_data = np.array(trans_data)

    return trans_data.reshape(len(data), feature_size*distribution)

def calLogicToQuantilization(data, distribution):
    f_size = int(len(data[0])/distribution)
    logic = np.zeros((len(data), f_size, distribution))
    for i, d in enumerate(data):
        d = np.array(d).reshape(f_size, distribution)
        for j, f in enumerate(d):
            logic[i][j][np.where(f == max(f))] = 1
    return logic.reshape(len(data), len(data[0]))

#########################################################################################################