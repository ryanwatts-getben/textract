from requests_html import HTMLSession
from bs4 import BeautifulSoup
import json
import pyppeteer
from playwright.sync_api import sync_playwright
import time
import sys

def scrape_endpoint(url):
    print(f"Scraping endpoint: {url}")
    errors = []
    api_info = {
        'verb': 'GET',
        'endpoint_name': None,
        'endpoint_url': None,
        'path_parameters': {},
        'query_parameters': {},
        'headers': {},
        'request_body': "",
        'request_example': "",
        'response_example': ""  # New field for response example
    }

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url)
            print("Waiting for JsonSchemaViewer...")
            page.wait_for_selector("div.JsonSchemaViewer")
            time.sleep(2)
            html_content = page.content()
            browser.close()
        print("Page loaded successfully")

        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract the endpoint name
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.text.strip()
            api_info['endpoint_name'] = title_text.split('|')[0].strip()
            print(f"Endpoint name: {api_info['endpoint_name']}")
        else:
            print("Warning: Endpoint name not found")
            errors.append("Endpoint name not found")

        # Extract the endpoint URL
        divs_with_title = soup.find_all('div', {'title': True})
        for div in divs_with_title:
            if "https://api.filevineapp.com/fv-app/v2" in div['title']:
                api_info['endpoint_url'] = div['title']
                print(f"Endpoint URL: {api_info['endpoint_url']}")
                break
        if not api_info['endpoint_url']:
            print("Warning: Endpoint URL not found")
            errors.append("Endpoint URL not found")

        # Extract path parameters
        print("Extracting path parameters...")
        path_params_section = soup.find('a', {'href': '#Path-Parameters'})
        if path_params_section:
            path_params_div = path_params_section.find_next('div', class_='JsonSchemaViewer')
            if path_params_div:
                params = path_params_div.find_all('div', {'data-test': lambda x: x and x.startswith('schema-row')})
                for param in params:
                    name_tag = param.find('div', {'data-test': lambda x: x and x.startswith('property-name-')})
                    type_tag = param.find('span', {'data-test': 'property-type'})
                    required_tag = param.find('span', {'data-test': 'property-required'})
                    description_tag = param.find('div', {'data-test': 'property-description'})

                    name = name_tag.text.strip() if name_tag else ''
                    type_ = type_tag.text.strip() if type_tag else ''
                    required = bool(required_tag)
                    description = description_tag.text.strip() if description_tag else ''

                    api_info['path_parameters'][name] = {
                        'type': type_,
                        'required': required,
                        'description': description
                    }
                print(f"Found {len(api_info['path_parameters'])} path parameters")
            else:
                print("Warning: Path parameters div not found")
                errors.append("Path parameters div not found")
        else:
            print("No path parameters section found")

        # Extract query parameters
        print("Extracting query parameters...")
        query_params_section = soup.find('a', {'href': '#Query-Parameters'})
        if query_params_section:
            query_params_div = query_params_section.find_next('div', class_='JsonSchemaViewer')
            if query_params_div:
                params = query_params_div.find_all('div', {'data-test': lambda x: x and x.startswith('schema-row')})
                for param in params:
                    name_tag = param.find('div', {'data-test': lambda x: x and x.startswith('property-name-')})
                    type_tag = param.find('span', {'data-test': 'property-type'})
                    required_tag = param.find('span', {'data-test': 'property-required'})
                    description_tag = param.find('div', {'data-test': 'property-description'})

                    name = name_tag.text.strip() if name_tag else ''
                    type_ = type_tag.text.strip() if type_tag else ''
                    required = bool(required_tag)
                    description = description_tag.text.strip() if description_tag else ''

                    api_info['query_parameters'][name] = {
                        'type': type_,
                        'required': required,
                        'description': description
                    }
                print(f"Found {len(api_info['query_parameters'])} query parameters")
            else:
                print("Warning: Query parameters div not found")
                errors.append("Query parameters div not found")
        else:
            print("No query parameters section found")

        # Extract headers
        print("Extracting headers...")
        headers_section = soup.find('a', {'href': '#request-headers'})
        if headers_section:
            headers_div = headers_section.find_next('div', class_='JsonSchemaViewer')
            if headers_div:
                header_rows = headers_div.find_all('div', {'data-test': lambda x: x and x.startswith('schema-row')})
                for header in header_rows:
                    name_tag = header.find('div', {'data-test': lambda x: x and x.startswith('property-name-')})
                    type_tag = header.find('span', {'data-test': 'property-type'})
                    required_tag = header.find('span', {'data-test': 'property-required'})
                    description_tag = header.find('div', {'data-test': 'property-description'})

                    name = name_tag.text.strip() if name_tag else ''
                    type_ = type_tag.text.strip() if type_tag else ''
                    required = bool(required_tag)
                    description = description_tag.text.strip() if description_tag else ''

                    api_info['headers'][name] = {
                        'type': type_,
                        'required': required,
                        'description': description
                    }
                print(f"Found {len(api_info['headers'])} headers")
            else:
                print("Warning: Headers div not found")
                errors.append("Headers div not found")
        else:
            print("No headers section found")

        # Extract request body
        print("Extracting request body...")
        request_body_textarea = soup.find('textarea', {'class': 'npm__react-simple-code-editor__textarea'})
        if request_body_textarea:
            api_info['request_body'] = request_body_textarea.text.strip()
            print("Request body found")
        else:
            print("Warning: Request body not found")
            errors.append("Request body not found")

        # Extract 2nd request example
        print("Extracting 2nd request example...")
        request_div = soup.find('div', {'class': 'sl-panel__content sl-p-0'})
        if request_div:
            request_pre = request_div.find('pre')
            if request_pre:
                api_info['request_example'] = request_pre.text.strip()
                print("request example found")
            else:
                print("Warning: request example not found")
                errors.append("request example not found")
        else:
            print("Warning: request div not found")
            errors.append("request div not found")

        # Extract response example
        print("Extracting response example...")
        response_div = soup.find('div', {'class': 'sl-code-highlight prism-code language-json'})
        if response_div:
            api_info['response_example'] = response_div.text.strip()
            print("Response example found")
        else:
            print("Warning: Response example div not found")
            errors.append("Response example div not found")

    except Exception as e:
        print(f"Error during scraping: {str(e)}")
        errors.append(f"Scraping error: {str(e)}")

    return api_info, errors

def main():
    base_url = 'https://developer.filevine.io'
    urls = [
        '/docs/v2-us/642f02a8dce06-get-project-fund-balance',
'/docs/v2-us/09ed3db47c825-create-project-funds-transaction',
'/docs/v2-us/bdbb8b9d9e1fb-get-project-fund-transaction',
'/docs/v2-us/47ab95b41e24c-voids-a-project-fund',
'/docs/v2-us/055697f1f6d87-get-data-connector-jobs',
'/docs/v2-us/bbad731cb4c6d-enqueues-a-data-connector-job',
'/docs/v2-us/6799e83748d39-get-data-connector-job',
'/docs/v2-us/813beea5c0473-create-data-connector-pickup-url',
'/docs/v2-us/4e6947baf52fa-get-stored-image',
'/docs/v2-us/f0eabe24b6762-assign-teams-to-projects',
'/docs/v2-us/f5c06bf8e8cf7-gets-a-list-of-teams',
'/docs/v2-us/5eb70b58ad4d0-create-a-team-in-an-org',
'/docs/v2-us/7de35b0b5b65c-get-details-of-a-team',
'/docs/v2-us/d6f03b4ebad60-adds-team-members-to-a-specific-team',
'/docs/v2-us/0d6ccba069ae7-adds-roles-to-team-members',
'/docs/v2-us/73b9bef550d9b-remove-team-members-from-a-team',
'/docs/v2-us/44f6693054b28-adds-a-team-to-a-project',
'/docs/v2-us/ec5d4e104d5e5-removes-a-team-from-a-project',
'/docs/v2-us/11f65c09c50db-get-project-team',
'/docs/v2-us/e60ced97049a6-add-project-team-member',
'/docs/v2-us/5279867593e3b-deprecated-get-project-roles',
'/docs/v2-us/8c66f60f07a69-get-project-org-roles',
'/docs/v2-us/4c9193e500341-get-project-team-org-roles-with-members-and-positions',
'/docs/v2-us/2155fb55f6e87-assign-roles-to-member',
'/docs/v2-us/9717a13dabe31-get-project-team-member',
'/docs/v2-us/d7cbe444cff42-remove-project-team-member',
'/docs/v2-us/a1996a2bdeb3c-update-project-team-member',
'/docs/v2-us/e2de649a5dcab-get-project-teams',
'/docs/v2-us/71cb32479ab00-import-deadline-chain-definition',
'/docs/v2-us/260dbec845d72-exports-deadline-chain-definition',
'/docs/v2-us/85f27a90429b2-imports-saved-report-definition',
'/docs/v2-us/ddd44431564d0-exports-saved-report-definition',
'/docs/v2-us/26e1e6d5728b6-import-custom-project-type-definition',
'/docs/v2-us/1c8d9cd1e3ba0-exports-custom-project-type-definition',
'/docs/v2-us/33ddf9a98c238-import-custom-section-definition',
'/docs/v2-us/97a3d3d649874-export-custom-section-definition',
'/docs/v2-us/0d34654615b6e-export-custom-project-type-definition',
'/docs/v2-us/4f13e0319ab87-get-contact-metadata',
'/docs/v2-us/e7b26a387bd09-create-custom-contact',
'/docs/v2-us/87805ab1c7116-update-custom-contact',
'/docs/v2-us/d4c35f916bc5c-get-custom-contact-tab',
'/docs/v2-us/931705c5f90aa-get-appointment',
'/docs/v2-us/0f53040bd0a04-update-appointment',
'/docs/v2-us/0d1faaf13dc09-delete-appointment',
'/docs/v2-us/bec0648917762-get-project-appointment-list',
'/docs/v2-us/8d2f454a014d1-create-project-appointment',
'/docs/v2-us/52b3731177ba0-get-contact-list',
'/docs/v2-us/8c31214637297-create-contact',
'/docs/v2-us/cd383c39729dc-get-contact',
'/docs/v2-us/8a829ba2b55de-update-contact',
'/docs/v2-us/ea9eebf01fbdb-get-projects-for-a-contact',
'/docs/v2-us/2976c3ae4c264-get-document-list',
'/docs/v2-us/c5009ec0c1d04-create-document-url-for-upload',
'/docs/v2-us/2536e9c2d3c86-get-document-series',
'/docs/v2-us/31a9394ed474a-get-document-series-metadata',
'/docs/v2-us/5bb2fe0d47fbe-get-document',
'/docs/v2-us/f3fdbeb90de1d-update-document-metadata',
'/docs/v2-us/7064529457906-delete-document',
'/docs/v2-us/ea82f83ec4887-get-document-download-locator',
'/docs/v2-us/598610ed45f8a-add-document-revision',
'/docs/v2-us/22efb603766f6-batch-document-download',
'/docs/v2-us/a86c8ea361b9a-batch-document-upload',
'/docs/v2-us/9faa4f1ad3b65-batch-document-upload-confirmation',
'/docs/v2-us/782be60c82501-create-a-fv-app-api-v-2-document',
'/docs/v2-us/32989758d1eb7-add-document-to-project',
'/docs/v2-us/aa956fe1d6c64-deprecated-get-project-document-list',
'/docs/v2-us/b776675ea442a-get-folder-list',
'/docs/v2-us/52792f2719113-create-folder',
'/docs/v2-us/1a832c3ee5b11-get-folder-structure',
'/docs/v2-us/0b31561da64a8-get-folder',
'/docs/v2-us/fc929157224eb-update-folder',
'/docs/v2-us/781afe3533fa5-delete-folder',
'/docs/v2-us/7e0e48205b35e-get-folder-children',
'/docs/v2-us/710b2708d647c-get-deadline-chain-type-list',
'/docs/v2-us/a4dd2525ca0a8-get-comment-list',
'/docs/v2-us/79b4ddbb72ec3-create-comment',
'/docs/v2-us/9a694a57f2dc3-get-comment',
'/docs/v2-us/59b0ecd2c774f-update-comment',
'/docs/v2-us/21f6956f7f338-get-note-feed-for-user',
'/docs/v2-us/da00bc90e087f-create-note',
'/docs/v2-us/7c65168c5df29-get-note',
'/docs/v2-us/a7006d2ee6600-update-note',
'/docs/v2-us/6c98cfe0d2b05-get-partner-id',
'/docs/v2-us/d841f0fd6991e-update-partner-id',
'/docs/v2-us/5a8faa0a54711-delete-partner-id',
'/docs/v2-us/3bf29e87af677-get-project-collection-item-list',
'/docs/v2-us/a517d24a9ce13-create-collection-item',
'/docs/v2-us/5ec97bd91210a-get-collection-item',
'/docs/v2-us/c6c86edda2ed2-update-collection-item',
'/docs/v2-us/3133f111934e6-delete-collection-item',
'/docs/v2-us/75f2252b0bd9b-add-contacts-to-project',
'/docs/v2-us/70b5078aa4682-get-project-contact-list',
'/docs/v2-us/28a1b51f5e828-update-project-contact',
'/docs/v2-us/ca39cda00eedb-delete-project-contact',
'/docs/v2-us/57e8ad20594ed-create-project-deadline-chain',
'/docs/v2-us/03108fee6e0a5-get-deadline-chain-list',
'/docs/v2-us/4bc00e8cd8b29-get-deadline-chain',
'/docs/v2-us/2152b399efb45-delete-deadline-chain',
'/docs/v2-us/42a44185b6818-update-deadline-chain',
'/docs/v2-us/d04e13a0a958d-update-deadline-chain-date',
'/docs/v2-us/830c229b93255-create-project-deadline',
'/docs/v2-us/d6a77259cd11e-get-project-deadline-list',
'/docs/v2-us/4fdaeea358949-get-project-deadline',
'/docs/v2-us/98e76b216c04f-update-project-deadline',
'/docs/v2-us/6361743b07257-delete-project-deadline',
'/docs/v2-us/9a615365a761f-get-form',
'/docs/v2-us/6a7ab25c342fb-update-form',
'/docs/v2-us/cc034280244f7-get-project-note-list',
'/docs/v2-us/fd0cd4ff78463-add-email-to-project',
'/docs/v2-us/b7076deb28f6b-get-project-email-list',
'/docs/v2-us/4c004ea45bd3f-get-project-task-list',
'/docs/v2-us/7803d492558cf-get-project-type-list',
'/docs/v2-us/f257b8b90cdee-get-project-type',
'/docs/v2-us/6a38a5c808dbb-get-project-type-sections-list',
'/docs/v2-us/02fa95af2a400-get-project-type-section',
'/docs/v2-us/45f47cc99a446-get-project-type-phase-list',
'/docs/v2-us/827459264c6f9-get-saved-reports-list',
'/docs/v2-us/4b5d2424e0295-run-saved-report',
'/docs/v2-us/d5de833c5cd45-get-tasks-for-user',
'/docs/v2-us/d4952175a619e-create-task',
'/docs/v2-us/bf821d27cb783-get-task',
'/docs/v2-us/02a42a9eb5586-update-task-body',
'/docs/v2-us/29c304eb66bdb-unassign-task',
'/docs/v2-us/376947ad57f90-assign-task',
'/docs/v2-us/9e3c4d94d3bff-complete-task',
'/docs/v2-us/3593babddff1f-uncomplete-task',
'/docs/v2-us/f86033865d7e3-change-task-due-date',
'/docs/v2-us/7b722e5eaf6ac-create-user',
'/docs/v2-us/006d0bda075f1-get-user-list',
'/docs/v2-us/4ed5d62613947-get-api-user',
'/docs/v2-us/1e435f79fbe45-get-user',
'/docs/v2-us/306bbbf7bc836-remove-user',
'/docs/v2-us/9776f0dc0df60-get-user-s-recent-projects',
'/docs/v2-us/d04c70293bc86-get-user-s-calendar-items',
'/docs/v2-us/7bd09e1e16fa3-get-user-orgs-with-token',
'/docs/v2-us/a9590f13c6d88-used-by-external-e2e-tests-to-check-api-v2-authentication',
'/docs/v2-us/5af577edf13af-mass-update-phase',
'/docs/v2-us/55cf2a603f578-mass-update-deadlines'
    ]

    results = []
    all_errors = []

    for relative_url in urls:
        full_url = base_url + relative_url
        print(f"\nProcessing URL: {full_url}")
        api_info, endpoint_errors = scrape_endpoint(full_url)
        results.append(api_info)
        if endpoint_errors:
            all_errors.append(f"Errors for {full_url}:")
            all_errors.extend(endpoint_errors)

    # Append results to scraper-backup.txt
    print("\nWriting results to scraper-backup.txt")
    with open('scraper-backup.txt', 'a') as f:
        json.dump(results, f, indent=2)

    # Write errors to scraper-backup-errors.txt
    if all_errors:
        print("Writing errors to scraper-backup-errors.txt")
        with open('scraper-backup-errors.txt', 'a') as f:
            for error in all_errors:
                f.write(error + '\n')
    else:
        print("No errors to report")

if __name__ == "__main__":
    main()
