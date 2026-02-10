#!/usr/bin/env python3
import json

with open('coverage.json') as f:
    data = json.load(f)

# Get coverage breakdown by files
files = data['data'][0]['files']

# Filter and sort files by line coverage
files_with_coverage = []
for f in files:
    summary = f.get('summary', {})
    lines = summary.get('lines', {})
    if lines.get('count', 0) > 0:
        files_with_coverage.append((f, lines))

# Sort by coverage percent (lowest first)
files_sorted = sorted(files_with_coverage, key=lambda x: x[1].get('percent', 0))

print('=== Coverage Report - Lines (Lowest First) ===')
print('')
print(f"{'File':<55} {'Covered':>10} {'Total':>10} {'%':>8}")
print('-' * 90)

for f, lines in files_sorted[:40]:  # Show lowest 40
    name = f['filename']
    if '\\' in name:
        name = name.split('\\')[-1]
    covered = lines['covered']
    total = lines['count']
    percent = lines['percent']
    print(f'{name:<55} {covered:>10} {total:>10} {percent:>7.2f}%')

print('')
print('='*90)
totals = data['data'][0]['totals']
print(f"{'TOTAL':<55} {totals['lines']['covered']:>10} {totals['lines']['count']:>10} {totals['lines']['percent']:>7.2f}%")

# Functions coverage
print('')
print('=== Functions Coverage (Lowest First) ===')
print('')
print(f"{'File':<55} {'Covered':>10} {'Total':>10} {'%':>8}")
print('-' * 90)

func_files = []
for f in files:
    summary = f.get('summary', {})
    funcs = summary.get('functions', {})
    if funcs.get('count', 0) > 0:
        func_files.append((f, funcs))

func_sorted = sorted(func_files, key=lambda x: x[1].get('percent', 0))
for f, funcs in func_sorted[:20]:
    name = f['filename']
    if '\\' in name:
        name = name.split('\\')[-1]
    covered = funcs['covered']
    total = funcs['count']
    percent = funcs['percent']
    print(f'{name:<55} {covered:>10} {total:>10} {percent:>7.2f}%')

print('')
print('='*90)
print(f"{'TOTAL':<55} {totals['functions']['covered']:>10} {totals['functions']['count']:>10} {totals['functions']['percent']:>7.2f}%")

# Branches coverage
print('')
print('=== Branches Coverage (Lowest First) ===')
print('')
print(f"{'File':<55} {'Covered':>10} {'Total':>10} {'%':>8}")
print('-' * 90)

branch_files = []
for f in files:
    summary = f.get('summary', {})
    branches = summary.get('branches', {})
    if branches.get('count', 0) > 0:
        branch_files.append((f, branches))

branch_sorted = sorted(branch_files, key=lambda x: x[1].get('percent', 0))
for f, branches in branch_sorted[:20]:
    name = f['filename']
    if '\\' in name:
        name = name.split('\\')[-1]
    covered = branches['covered']
    total = branches['count']
    percent = branches['percent']
    print(f'{name:<55} {covered:>10} {total:>10} {percent:>7.2f}%')

if branch_files:
    print('')
    print('='*90)
    print(f"{'TOTAL':<55} {totals['branches']['covered']:>10} {totals['branches']['count']:>10} {totals['branches']['percent']:>7.2f}%")
else:
    print('No branch coverage data available')

# Modules with <100% coverage
print('')
print('=== Files with <100% Line Coverage ===')
print('')
low_coverage = [(f, lines) for f, lines in files_sorted if lines['percent'] < 100.0]
if low_coverage:
    print(f"{'File':<55} {'%':>8} {'Missing':>10}")
    print('-' * 80)
    for f, lines in low_coverage:
        name = f['filename']
        if '\\' in name:
            name = name.split('\\')[-1]
        missing = lines['count'] - lines['covered']
        pct = lines['percent']
        print(f'{name:<55} {pct:>7.2f}% {missing:>10}')
else:
    print('All files have 100% line coverage!')
