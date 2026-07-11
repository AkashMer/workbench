import argparse
import sys
import yaml

def main():
    # Define the arguments
    parser = argparse.ArgumentParser()
    # The file to be preserved
    parser.add_argument('--environment', required=True)
    # The file to serve as the lookup
    parser.add_argument('--full', required=True)
    # Output file name
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    # Load both ymls
    with open(args.environment, 'r') as f:
        history_env = yaml.safe_load(f)
    with open(args.full, 'r') as f:
        full_env = yaml.safe_load(f)

    # Build the additions
    lookup = {}
    pip_section = None
    for entry in full_env['dependencies']:
        # Grab the versions of overlapping libraries
        if isinstance(entry, str):
            name, _, version = entry.partition('=')
            lookup[name] = version
        # Grab the pip section as is
        elif isinstance(entry, dict) and 'pip' in entry:
            pip_section = entry

    # Add the found additions
    merged_deps = []
    for entry in history_env['dependencies']:
        # Add the version numbers
        if isinstance(entry, str):
            name = entry.partition('=')[0]
            version = lookup.get(name)
            if version:
                merged_deps.append(f'{name}={version}')
            else:
                print(f'WARNING: no version found for "{name}", leaving unpinned', file=sys.stderr)
                merged_deps.append(name)
        else:
            merged_deps.append(entry)

    # Add the pip section
    if pip_section and not any(isinstance(e, dict) and 'pip' in e for e in merged_deps):
        merged_deps.append(pip_section)

    # Build the return dict
    result = {
        'name': history_env['name'],
        'channels': full_env['channels'], # Make sure library source channels are preserved
        'dependencies': merged_deps,
    }

    # Save it to the file to be preserved
    with open(args.out, 'w') as f:
        yaml.safe_dump(result, f, sort_keys=False, default_flow_style=False)

# Call the main function
if __name__ == '__main__':
    main()
