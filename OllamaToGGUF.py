import os
import json
import shutil
import sys

# Determine the current directory where the script is running
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define directory variables
manifest_dir = os.path.join(os.path.expanduser('~'), '.ollama', 'models', 'manifests', 'registry.ollama.ai')
blob_dir = os.path.join(os.path.expanduser('~'), '.ollama', 'models', 'blobs')
outputModels_dir = os.path.join(current_dir, 'Output')  # Updated to use the current directory

# Print base directories to confirm variables
print("\nOllama To GGUF\n")
print("Confirming Directories:")
print(f"Manifest Directory: {manifest_dir}")
print(f"Blob Directory: {blob_dir}")
print(f"Output Models Directory: {outputModels_dir}")

# Ensure public models directory exists or create it
if not os.path.exists(outputModels_dir):
    os.makedirs(outputModels_dir)
    print("Output Models Directory Created.")
else:
    print("Output Models Directory Confirmed.")

# Explore manifest directory and record manifest file locations
files = []
for dirpath, _, filenames in os.walk(manifest_dir):
    for filename in filenames:
        files.append(os.path.join(dirpath, filename))
manifest_locations = files[:]
if not manifest_locations:
    print("No manifest files found.")
else:
    print("Manifest files found")

def get_model_size(layers, blob_directory):
    total_size = 0
    for layer_info in layers:
        digest = layer_info['digest']
        sha = digest.split(':')[1]
        full_sha = f'sha256-{sha}'
        source_blob = os.sep.join([blob_directory, str(full_sha)])
        if os.path.exists(source_blob):
            total_size += os.path.getsize(source_blob)
    return total_size

def recombine_model(manifest_path, blob_directory, output_directory):
    """
    Recombine model parts from Ollama's split format into a single GGUF equivalent file.
    
    :param manifest_path: Path to the JSON manifest file describing the model.
    :param blob_directory: Path where all blob parts are stored.
    :param output_directory: Target directory where combined gguf will be saved.
    
    :return Boolean: Success or failure
    """
    
    # Load and parse JSON data from manifest
    with open(manifest_path) as f_obj:
        obj = json.load(f_obj)
    
    config = obj.get('config')
    
    if not config:
        raise ValueError('Config section missing from JSON.')
    
    digest = config.get('digest')
    
    if not digest:
        raise ValueError('Digest missing from config section.')
    
    sha_value = digest.split(':')[-1]
     
    # Prepare paths based on SHA value extracted
    sha256_value = f'sha256-{sha_value}'
    sha_file = os.sep.join([blob_directory, str(sha256_value)])
     
    # Load configuration data about this specific SHA value
    with open(sha_file) as f_model_config_obj:
        config_data = json.load(f_model_config_obj)
     
    try:
        modelQuant = config_data['file_type']
        assert len(modelQuant) > 0, "Model quantization type cannot be empty."
        assert isinstance(modelQuant, str), "Model quantization type must be string."
    except Exception as e:
        raise ValueError("Invalid or missing `file_type` parameter.") from e
    
    try:
        trained_on_key = 'model_type'
        trained_on_default = 'unknown'
        trained_on = str(config_data[trained_on_key])
    except KeyError as e:
        trained_on = trained_on_default
    
    layers = obj.get('layers')
    
    if not layers:
        raise ValueError("Layers section is required but missing.")
    
    modelName = os.path.basename(os.path.dirname(manifest_path))
    target_subdir = os.sep.join([output_directory, modelName])
    combined_filename = f"{modelName}-{trained_on}-{modelQuant}.gguf"
    
    # Check for large files and use chunked processing if needed
    large_file_threshold = 1 * 1024 * 1024 * 1024  # 1 GB
    total_size = get_model_size(layers, blob_directory)
    
    if total_size > large_file_threshold:
        print(f"Model size is large ({total_size / (1024*1024*1024):.2f} GB). Using chunked processing.")
        return recombine_model_chunked(manifest_path, blob_directory, output_directory, layers, 
                                      modelName, target_subdir, combined_filename)
    
    # For smaller models, use regular in-memory approach
    print("Using standard processing")
    layer_contents = []
    print("reading layers")
    errors_occurred = False
    
    for layer_index, layer_info in enumerate(layers):
        print("Layer Index: {0}".format(layer_index))
        mediaType = layer_info['mediaType']
        digest = layer_info['digest']
        sha = digest.split(':')[1]
        full_sha = f'sha256-{sha}'
        source_blob = os.sep.join([blob_directory, str(full_sha)])
        
        prefix = f'\t[{modelName}] [{mediaType}]'
        
        try:
            if not os.path.exists(source_blob):
                raise FileNotFoundError(f"Blob file not found: {source_blob}")
                
            status = "Reading"
            msg = (prefix + f'\tBlob SHA File Path : [{source_blob}] ')
            sys.stdout.write((msg.ljust(80) + status.rjust(48) + "\r\n"))
            
            with open(source_blob, 'rb') as layer_fobj:
                content = layer_fobj.read()
                layer_contents.append(content)
            
            status = "Success"
            sys.stdout.write((prefix + f'\tRead blob successfully: {len(content)} bytes').ljust(80) + status.rjust(48) + "\r\n")
                
        except Exception as excp:
            errors_occurred = True
            status = "Failed"
            error_message = f"Failed reading [{source_blob}]: {str(excp)}"
            sys.stdout.write((prefix + f'\t{error_message}').ljust(80) + status.rjust(48) + "\r\n")
    
    # Only proceed to write the output file if all layers were read successfully
    if errors_occurred:
        print(f"Errors occurred while processing layers. Conversion for {modelName} aborted.")
        return False
    
    if not layer_contents:
        print(f"No layers were successfully read for {modelName}. Conversion aborted.")
        return False
    
    # Create combined content and write to file
    try:
        combined_content = b''.join(layer_contents)
        
        if not os.path.exists(target_subdir):
            os.makedirs(target_subdir)
        
        final_output_filepath = os.sep.join([target_subdir, combined_filename])
        
        with open(final_output_filepath, 'wb') as final_fobj:
            final_fobj.write(combined_content)
            
        print(f"Successfully created {final_output_filepath} ({len(combined_content) / (1024 * 1024):.2f} MB)")
        return True
        
    except Exception as excp:
        print(f"Error writing output file: {str(excp)}")
        return False

def recombine_model_chunked(manifest_path, blob_directory, output_directory, layers, 
                           modelName, target_subdir, combined_filename, chunk_size=16*1024*1024):
    """
    Recombine model using chunk-based processing for large files.
    
    :param chunk_size: Size of chunks to process at once (default 16MB)
    :return: Boolean indicating success or failure
    """
    print(f"Processing large model in chunks of {chunk_size/(1024*1024):.0f}MB")
    
    if not os.path.exists(target_subdir):
        os.makedirs(target_subdir)
    
    final_output_filepath = os.sep.join([target_subdir, combined_filename])
    
    try:
        with open(final_output_filepath, 'wb') as final_fobj:
            for layer_index, layer_info in enumerate(layers):
                print("Layer Index: {0}".format(layer_index))
                mediaType = layer_info['mediaType']
                digest = layer_info['digest']
                sha = digest.split(':')[1]
                full_sha = f'sha256-{sha}'
                source_blob = os.sep.join([blob_directory, str(full_sha)])
                
                prefix = f'\t[{modelName}] [{mediaType}]'
                
                if not os.path.exists(source_blob):
                    print(f"{prefix}\tBLob file not found: {source_blob}")
                    return False
                
                try:
                    # Get file size to report progress
                    file_size = os.path.getsize(source_blob)
                    bytes_processed = 0
                    
                    status = "Reading"
                    msg = (prefix + f'\tProcessing blob: [{source_blob}] ({file_size/(1024*1024):.2f} MB)')
                    sys.stdout.write((msg.ljust(80) + status.rjust(48) + "\r\n"))
                    
                    with open(source_blob, 'rb') as layer_fobj:
                        while True:
                            chunk = layer_fobj.read(chunk_size)
                            if not chunk:
                                break
                            final_fobj.write(chunk)
                            bytes_processed += len(chunk)
                            progress = int(bytes_processed / file_size * 100)
                            sys.stdout.write(f"\r\t\tProgress: {progress}% ({bytes_processed/(1024*1024):.2f} MB / {file_size/(1024*1024):.2f} MB)")
                            sys.stdout.flush()
                    
                    sys.stdout.write("\n")
                    status = "Success"
                    sys.stdout.write((prefix + f'\tProcessed blob: {file_size} bytes').ljust(80) + status.rjust(48) + "\r\n")
                    
                except Exception as excp:
                    status = "Failed"
                    error_message = f"Failed processing [{source_blob}]: {str(excp)}"
                    sys.stdout.write((prefix + f'\t{error_message}').ljust(80) + status.rjust(48) + "\r\n")
                    return False
        
        final_size = os.path.getsize(final_output_filepath)
        print(f"Successfully created {final_output_filepath} ({final_size / (1024 * 1024):.2f} MB)")
        return True
        
    except Exception as excp:
        print(f"Error during chunked processing: {str(excp)}")
        # Clean up partial file if it exists
        if os.path.exists(final_output_filepath):
            try:
                os.remove(final_output_filepath)
                print(f"Removed incomplete output file.")
            except:
                pass
        return False

def main():
    while True:
        if not manifest_locations:
            print("No manifest files found.")
            break

        print("\nAvailable Ollama Models to Convert:\n")
        for index, manifest_path in enumerate(manifest_locations, start=1):
            modelName = os.path.basename(os.path.dirname(manifest_path))
            manifest_filename = os.path.basename(manifest_path)
            
            # Load the manifest file to get the quantization type and layers
            with open(manifest_path) as f_obj:
                obj = json.load(f_obj)
            
            config = obj.get('config')
            if not config:
                print(f"{index}. {modelName} (Manifest: {manifest_filename}, Quantization: Unknown, Size: Unknown)")
                continue
            
            digest = config.get('digest')
            if not digest:
                print(f"{index}. {modelName} (Manifest: {manifest_filename}, Quantization: Unknown, Size: Unknown)")
                continue
            
            sha_value = digest.split(':')[-1]
            sha256_value = f'sha256-{sha_value}'
            sha_file = os.sep.join([blob_dir, str(sha256_value)])
            
            # Load configuration data about this specific SHA value
            try:
                with open(sha_file) as f_model_config_obj:
                    config_data = json.load(f_model_config_obj)
                
                modelQuant = config_data.get('file_type', 'Unknown')
            except Exception as e:
                modelQuant = 'Unknown'
            
            layers = obj.get('layers', [])
            model_size = get_model_size(layers, blob_dir)
            model_size_str = f"{model_size / (1024 * 1024):.2f} MB" if model_size > 0 else "Unknown"
            
            print(f"{index}. {modelName} (Manifest: {manifest_filename}, Quantization: {modelQuant}, Size: {model_size_str})")
        
        try:
            choice = int(input("\nEnter the number of the model you want to convert (or 0 to exit): "))
            if choice == 0:
                print("Exiting.")
                break
            elif 1 <= choice <= len(manifest_locations):
                manifest_path = manifest_locations[choice - 1]
                success = recombine_model(manifest_path, blob_dir, outputModels_dir)
                if success:
                    print(f"Successfully converted {os.path.basename(os.path.dirname(manifest_path))} to GGUF.")
                else:
                    print(f"Conversion failed for {os.path.basename(os.path.dirname(manifest_path))}.")
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()