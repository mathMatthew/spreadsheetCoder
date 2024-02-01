def ask_question_validation_function(prompt, is_valid):
    attempts = 0
    max_attempts = 3

    while True:
        response = input(f"{prompt}: ")
        if is_valid(response):
            return response
        else:
            attempts += 1
            if attempts < max_attempts:
                remaining_attempts = max_attempts - attempts
                print(
                    f"Invalid response. Attempts remaining: {remaining_attempts}. Please try again."
                )
            else:
                continue_response = input(
                    "You have exceeded the maximum number of attempts. Do you want to continue? (yes/no): "
                )
                if continue_response.lower() in ["yes", "y"]:
                    attempts = 0  # Reset attempts
                else:
                    print("Operation canceled.")
                    return None

def ask_question(prompt, valid_responses) -> str:
    while True:
        response = input(
            f"{prompt} \n Respond with {' or '.join(valid_responses)}: "
        ).lower()
        if response in valid_responses:
            return response
        else:
            print(
                f"Invalid response. Please answer with '{' or '.join(valid_responses)}'."
            )
