from pkg_resources import resource_string

# Loads in sweet title ASCII art
title = resource_string('taqtoe.resources', 'title.txt').decode('ascii')

if __name__ == '__main__':
    print(title)
