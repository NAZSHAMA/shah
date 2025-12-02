# Gemini Model

This document provides information about the Gemini model and related functions.

## `GeminiModel` Class

The `GeminiModel` class is used to interact with the Gemini model.

```python
class GeminiModel:
    """Class for the Gemini model."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-001",
        finetuned_model: bool = False,
        distribute_requests: bool = False,
        cache_name: str | None = None,
        temperature: float = 0.01,
        **kwargs,
    ):
        self.model_name = model_name
        self.finetuned_model = finetuned_model
        self.arguments = kwargs
        self.distribute_requests = distribute_requests
        self.temperature = temperature
        model_name = self.model_name
        if not self.finetuned_model and self.distribute_requests:
            random_region = random.choice(GEMINI_AVAILABLE_REGIONS)
            model_name = GEMINI_URL.format(
                GCP_PROJECT=GCP_PROJECT,
                region=random_region,
                model_name=self.model_name,
            )
        if cache_name is not None:
            cached_content = caching.CachedContent(cached_content_name=cache_name)
            self.model = GenerativeModel.from_cached_content(
                cached_content=cached_content
            )
        else:
            self.model = GenerativeModel(model_name=model_name)

    @retry(max_attempts=12, base_delay=2, backoff_factor=2)
    def call(self, prompt: str, parser_func=None) -> str:
        """Calls the Gemini model with the given prompt.

        Args:
            prompt (str): The prompt to call the model with.
            parser_func (callable, optional): A function that processes the LLM
              output. It takes the model"s response as input and returns the
              processed result.

        Returns:
            str: The processed response from the model.
        """
        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=self.temperature,
                **self.arguments,
            ),
            safety_settings=SAFETY_FILTER_CONFIG,
        ).text
        if parser_func:
            return parser_func(response)
        return response

    def call_parallel(
        self,
        prompts: List[str],
        parser_func: Optional[Callable[[str], str]] = None,
        timeout: int = 60,
        max_retries: int = 5,
    ) -> List[Optional[str]]:
        """Calls the Gemini model for multiple prompts in parallel using threads with retry logic.

        Args:
            prompts (List[str]): A list of prompts to call the model with.
            parser_func (callable, optional): A function to process each response.
            timeout (int): The maximum time (in seconds) to wait for each thread.
            max_retries (int): The maximum number of retries for timed-out threads.

        Returns:
            List[Optional[str]]:
            A list of responses, or None for threads that failed.
        """
        results = [None] * len(prompts)

        def worker(index: int, prompt: str):
            """Thread worker function to call the model and store the result with retries."""
            retries = 0
            while retries <= max_retries:
                try:
                    return self.call(prompt, parser_func)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"Error for prompt {index}: {str(e)}")
                    retries += 1
                    if retries <= max_retries:
                        print(f"Retrying ({retries}/{max_retries}) for prompt {index}")
                        time.sleep(1)  # Small delay before retrying
                    else:
                        return f"Error after retries: {str(e)}"

        # Create and start one thread for each prompt
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            future_to_index = {
                executor.submit(worker, i, prompt): i
                for i, prompt in enumerate(prompts)
            }

            for future in as_completed(future_to_index, timeout=timeout):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"Unhandled error for prompt {index}: {e}")
                    results[index] = "Unhandled Error"

        # Handle remaining unfinished tasks after the timeout
        for future in future_to_index:
            index = future_to_index[future]
            if not future.done():
                print(f"Timeout occurred for prompt {index}")
                results[index] = "Timeout"

        return results
```

## `gemini_to_json_schema` Function

The `gemini_to_json_schema` function converts a Gemini Schema object into a JSON Schema dictionary.

```python
def gemini_to_json_schema(gemini_schema: Schema) -> Dict[str, Any]:
  """Converts a Gemini Schema object into a JSON Schema dictionary.

  Args:
      gemini_schema: An instance of the Gemini Schema class.

  Returns:
      A dictionary representing the equivalent JSON Schema.

  Raises:
      TypeError: If the input is not an instance of the expected Schema class.
      ValueError: If an invalid Gemini Type enum value is encountered.
  """
  if not isinstance(gemini_schema, Schema):
    raise TypeError(
        f"Input must be an instance of Schema, got {type(gemini_schema)}"
    )

  json_schema_dict: Dict[str, Any] = {}

  # Map Type
  gemini_type = getattr(gemini_schema, "type", None)
  if gemini_type and gemini_type != Type.TYPE_UNSPECIFIED:
    json_schema_dict["type"] = gemini_type.lower()
  else:
    json_schema_dict["type"] = "null"

  # Map Nullable
  if getattr(gemini_schema, "nullable", None) == True:
    json_schema_dict["nullable"] = True

  # --- Map direct fields ---
  direct_mappings = {
      "title": "title",
      "description": "description",
      "default": "default",
      "enum": "enum",
      "format": "format",
      "example": "example",
  }
  for gemini_key, json_key in direct_mappings.items():
    value = getattr(gemini_schema, gemini_key, None)
    if value is not None:
      json_schema_dict[json_key] = value

  # String validation
  if gemini_type == Type.STRING:
    str_mappings = {
        "pattern": "pattern",
        "min_length": "minLength",
        "max_length": "maxLength",
    }
    for gemini_key, json_key in str_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Number/Integer validation
  if gemini_type in (Type.NUMBER, Type.INTEGER):
    num_mappings = {
        "minimum": "minimum",
        "maximum": "maximum",
    }
    for gemini_key, json_key in num_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Array validation (Recursive call for items)
  if gemini_type == Type.ARRAY:
    items_schema = getattr(gemini_schema, "items", None)
    if items_schema is not None:
      json_schema_dict["items"] = gemini_to_json_schema(items_schema)

    arr_mappings = {
        "min_items": "minItems",
        "max_items": "maxItems",
    }
    for gemini_key, json_key in arr_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Object validation (Recursive call for properties)
  if gemini_type == Type.OBJECT:
    properties_dict = getattr(gemini_schema, "properties", None)
    if properties_dict is not None:
      json_schema_dict["properties"] = {
          prop_name: gemini_to_json_schema(prop_schema)
          for prop_name, prop_schema in properties_dict.items()
      }

    obj_mappings = {
        "required": "required",
        "min_properties": "minProperties",
        "max_properties": "maxProperties",
        # Note: Ignoring 'property_ordering' as it's not standard JSON Schema
    }
    for gemini_key, json_key in obj_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Map anyOf (Recursive call for subschemas)
  any_of_list = getattr(gemini_schema, "any_of", None)
  if any_of_list is not None:
    json_schema_dict["anyOf"] = [
        gemini_to_json_schema(sub_schema) for sub_schema in any_of_list
    ]

  return json_schema_dict
```
