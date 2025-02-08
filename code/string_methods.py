# pylint: disable=missing-docstring

# Warning:
# - One line of code for each method
# - Just look in the doc for the right method of the String class!

def add_comma(a_string):
    """
    returns a copy of the string with every word separated by a comma
    example: add_comma("John Peter Jude") => "John, Peter, Jude"
    """
    # $CHALLENGIFY_BEGIN
    return ', '.join(a_string.split()) # or a_string.replace(' ', ', ')
    # $CHALLENGIFY_END

def belongs_to(a_string, a_word):
    """
    returns True if a_string contains a_word
    example: belongs_to("hey jude", "jude") => True
    """
    # $CHALLENGIFY_BEGIN
    return a_word in a_string
    # $CHALLENGIFY_END

def count_repetition(a_string, a_substring):
    """
    returns how many times a_substring occurs in a_string
    example: count_repetition("000123000123", "0") => 6
    """
    # $CHALLENGIFY_BEGIN
    return a_string.count(a_substring)
    # $CHALLENGIFY_END

def is_a_question(a_string):
    """
    returns True if a_string ends with a "?"
    example: is_a_question("How are you?") => True
    """
    # $CHALLENGIFY_BEGIN
    return a_string.endswith('?')
    # $CHALLENGIFY_END

def remove_surrounding_whitespaces(a_string):
    """
    returns a copy of the string with leading and trailing whitespaces removed
    example: remove_surrounding_whitespaces("  hey yo  ") => "hey yo"
    """
    # $CHALLENGIFY_BEGIN
    return a_string.strip()
    # $CHALLENGIFY_END

def replace(initial_string, old_letter, new_letter):
    """
    returns a copy of the string with the new letter replacing the old one
    example: replace("casanova", "a", "o") => "cosonovo"
    """
    # $CHALLENGIFY_BEGIN
    return initial_string.replace(old_letter, new_letter)
    # $CHALLENGIFY_END

def full_description_concatenation(first_name, last_name, age):
    """
    returns a sentence with the first_name and the last_name capitalized and
     the age using concatenation
    example: full_description_concatenation("john", "doe", 33) => "John Doe is 33"
    """
    # $CHALLENGIFY_BEGIN
    return first_name.capitalize() + ' ' + last_name.capitalize() + ' is ' + str(age)
    # $CHALLENGIFY_END

def full_description_formatting(first_name, last_name, age):
    """
    returns a sentence with the first_name and the last_name capitalized and
     the age using string interpolation
    example: full_description_formatting("john", "doe", 33) => "John Doe is 33"
    """
    # $CHALLENGIFY_BEGIN
    return f"{first_name.capitalize()} {last_name.capitalize()} is {age}"
    # $CHALLENGIFY_END
