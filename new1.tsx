"use client"

import { useState } from 'react'
import { Search, ChevronDown, ChevronUp, ChevronLeft, ChevronRight, Calendar, User, Hospital, UserPlus, Stethoscope, FileText, Pill, Info } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

const mockData = [
  {
    "Date": "2023-05-15",
    "General": [
      {
        "Medically Relevant Information That Is Present On All Pages": [
          {
            "Patient": "John Doe",
            "Medical Facility": "General Hospital",
            "Referred By": "Dr. Smith",
            "Referred To": "Dr. Johnson",
            "Name of Doctor": ["Dr. Johnson", "Dr. Lee"],
            "ICD10CM": ["J45.909", "I10"],
            "CPT Codes": ["99213", "94010"],
            "Rx": ["Albuterol", "Lisinopril"],
            "Other Medically Relevant Information": ["Patient reports increased shortness of breath"],
            "Daily Summary": "Patient visited for follow-up on asthma and hypertension. Medications adjusted."
          }
        ]
      }
    ],
    "Page Numbers That Contain The Following Information": [
      {"Referred By": [{"Dr. Smith": "1"}]},
      {"Referred To": [{"Dr. Johnson": "1"}]},
      {"Name of Doctor": [{"Dr. Johnson": "1,2"}, {"Dr. Lee": "2"}]},
      {"ICD10CM": [{"J45.909": "1"}, {"I10": "1"}]},
      {"CPT Codes": [{"99213": "1"}, {"94010": "2"}]},
      {"Rx": [{"Albuterol": "2"}, {"Lisinopril": "2"}]},
      {"Other Medically Relevant Information": [{"Patient reports increased shortness of breath": "1"}]}
    ]
  },
  {
    "Date": "2023-05-16",
    "General": [
      {
        "Medically Relevant Information That Is Present On All Pages": [
          {
            "Patient": "John Doe",
            "Medical Facility": "City Radiology Center",
            "Referred By": "Dr. Johnson",
            "Referred To": null,
            "Name of Doctor": ["Dr. Anderson"],
            "ICD10CM": ["S13.4XXA"],
            "CPT Codes": ["72040", "72050"],
            "Rx": [],
            "Other Medically Relevant Information": ["Cervical spine X-ray performed"],
            "Daily Summary": "Patient undergoes cervical spine X-ray following motor vehicle accident. No acute fractures or dislocations observed."
          }
        ]
      }
    ],
    "Page Numbers That Contain The Following Information": [
      {"Referred By": [{"Dr. Johnson": "1"}]},
      {"Referred To": []},
      {"Name of Doctor": [{"Dr. Anderson": "1,2"}]},
      {"ICD10CM": [{"S13.4XXA": "1"}]},
      {"CPT Codes": [{"72040": "1"}, {"72050": "1"}]},
      {"Rx": []},
      {"Other Medically Relevant Information": [{"Cervical spine X-ray performed": "1"}]}
    ]
  },
  {
    "Date": "2023-05-18",
    "General": [
      {
        "Medically Relevant Information That Is Present On All Pages": [
          {
            "Patient": "John Doe",
            "Medical Facility": "Wellness Physical Therapy",
            "Referred By": "Dr. Johnson",
            "Referred To": null,
            "Name of Doctor": ["Dr. Williams"],
            "ICD10CM": ["S13.4XXA", "M54.2"],
            "CPT Codes": ["97110", "97140"],
            "Rx": [],
            "Other Medically Relevant Information": ["Neck ROM exercises prescribed"],
            "Daily Summary": "Initial physical therapy session for cervical strain. Patient reports mild improvement in neck pain. Therapeutic exercises and manual therapy performed."
          }
        ]
      }
    ],
    "Page Numbers That Contain The Following Information": [
      {"Referred By": [{"Dr. Johnson": "1"}]},
      {"Referred To": []},
      {"Name of Doctor": [{"Dr. Williams": "1,2"}]},
      {"ICD10CM": [{"S13.4XXA": "1"}, {"M54.2": "1"}]},
      {"CPT Codes": [{"97110": "1"}, {"97140": "2"}]},
      {"Rx": []},
      {"Other Medically Relevant Information": [{"Neck ROM exercises prescribed": "2"}]}
    ]
  },
  {
    "Date": "2023-05-22",
    "General": [
      {
        "Medically Relevant Information That Is Present On All Pages": [
          {
            "Patient": "John Doe",
            "Medical Facility": "Wellness Physical Therapy",
            "Referred By": "Dr. Johnson",
            "Referred To": null,
            "Name of Doctor": ["Dr. Williams"],
            "ICD10CM": ["S13.4XXA", "M54.2"],
            "CPT Codes": ["97110", "97035"],
            "Rx": [],
            "Other Medically Relevant Information": ["Ultrasound therapy applied"],
            "Daily Summary": "Follow-up physical therapy session. Patient reports continued improvement. Therapeutic exercises performed and ultrasound therapy applied to cervical region."
          }
        ]
      }
    ],
    "Page Numbers That Contain The Following Information": [
      {"Referred By": [{"Dr. Johnson": "1"}]},
      {"Referred To": []},
      {"Name of Doctor": [{"Dr. Williams": "1,2"}]},
      {"ICD10CM": [{"S13.4XXA": "1"}, {"M54.2": "1"}]},
      {"CPT Codes": [{"97110": "1"}, {"97035": "2"}]},
      {"Rx": []},
      {"Other Medically Relevant Information": [{"Ultrasound therapy applied": "2"}]}
    ]
  }
]

export default function Component() {
  const [visits, setVisits] = useState(mockData.sort((a, b) => new Date(a.Date).getTime() - new Date(b.Date).getTime()))
  const [selectedVisit, setSelectedVisit] = useState(visits[0])
  const [expandedVisits, setExpandedVisits] = useState<string[]>([])
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedItem, setSelectedItem] = useState<{type: string, value: string, pages: string} | null>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [initialVisitSelected, setInitialVisitSelected] = useState(false)

  const toggleVisitExpansion = (date: string) => {
    setExpandedVisits(prev => 
      prev.includes(date) ? prev.filter(d => d !== date) : [...prev, date]
    )
  }

  const handleItemClick = (type: string, value: string, pages: string) => {
    setSelectedItem({ type, value, pages })
    setCurrentPage(1)
    setInitialVisitSelected(false)
  }

  const handleSearch = (term: string) => {
    setSearchTerm(term)
    // Implement search logic here
  }

  const handleVisitSelect = (visit) => {
    setSelectedVisit(visit)
    setSelectedItem(null)
    setInitialVisitSelected(true)
  }

  const renderPageBadges = (pages: string) => {
    return pages.split(',').map((page, index) => (
      <Badge key={index} variant="secondary" className="mr-1">
        Page {page.trim()}
      </Badge>
    ))
  }

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Left sidebar - Timeline */}
      <div className="w-1/4 bg-white p-4 overflow-y-auto border-r">
        <Input
          type="search"
          placeholder="Search records..."
          className="mb-4"
          value={searchTerm}
          onChange={(e) => handleSearch(e.target.value)}
        />
        {visits.map((visit) => (
          <Card key={visit.Date} className="mb-2">
            <CardHeader className="p-2 cursor-pointer" onClick={() => toggleVisitExpansion(visit.Date)}>
              <CardTitle className="text-sm flex justify-between items-center">
                <span>{visit.Date}: {visit.General[0]["Medically Relevant Information That Is Present On All Pages"][0]["Medical Facility"]}, {visit.General[0]["Medically Relevant Information That Is Present On All Pages"][0]["Name of Doctor"].join(" and ")}</span>
                {expandedVisits.includes(visit.Date) ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </CardTitle>
            </CardHeader>
            {expandedVisits.includes(visit.Date) && (
              <CardContent className="p-2 text-xs">
                <Button 
                  variant="link" 
                  className="p-0 h-auto text-xs text-[#0084ff]" 
                  onClick={() => handleVisitSelect(visit)}
                >
                  View Details
                </Button>
              </CardContent>
            )}
          </Card>
        ))}
      </div>

      {/* Main content */}
      <div className="flex-1 p-4 overflow-y-auto">
        <h1 className="text-2xl font-bold mb-4">Medical Visit Details</h1>
        <Card>
          <CardContent className="p-4">
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p className="font-semibold flex items-center"><Calendar className="mr-2" size={16} /> Date: {selectedVisit.Date}</p>
                <p className="font-semibold flex items-center"><User className="mr-2" size={16} /> Patient: {selectedVisit.General[0]["Medically Relevant Information That Is Present On All Pages"][0].Patient}</p>
                <p className="font-semibold flex items-center"><Hospital className="mr-2" size={16} /> Facility: {selectedVisit.General[0]["Medically Relevant Information That Is Present On All Pages"][0]["Medical Facility"]}</p>
                <p className="font-semibold flex items-center"><UserPlus className="mr-2" size={16} /> Referred By: {selectedVisit.General[0]["Medically Relevant Information That Is Present On All Pages"][0]["Referred By"] || "N/A"}</p>
                <p className="font-semibold flex items-center"><Stethoscope className="mr-2" size={16} /> Referred To: {selectedVisit.General[0]["Medically Relevant Information That Is Present On All Pages"][0]["Referred To"] || "N/A"}</p>
              </div>
              <div>
                <p className="font-semibold">Attending Doctors:</p>
                <ul>
                  {selectedVisit.General[0]["Medically Relevant Information That Is Present On All Pages"][0]["Name of Doctor"].map((doctor, index) => (
                    <li key={index} className="cursor-pointer text-[#0084ff]" onClick={() => handleItemClick("Doctor", doctor, selectedVisit["Page Numbers That Contain The Following Information"][2]["Name of Doctor"].find(d => Object.keys(d)[0] === doctor)?.[doctor] || "")}>
                      {doctor} {renderPageBadges(selectedVisit["Page Numbers That Contain The Following Information"][2]["Name of Doctor"].find(d => Object.keys(d)[0] === doctor)?.[doctor] || "")}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p className="font-semibold">ICD-10 Codes:</p>
                <ul>
                  {selectedVisit.General[0]["Medically Relevant Information That Is Present On All Pages"][0]["ICD10CM"].map((code, index) => (
                    <li key={index} className="cursor-pointer text-[#0084ff]" onClick={() => handleItemClick("ICD10", code, selectedVisit["Page Numbers That Contain The Following Information"][3]["ICD10CM"].find(c => Object.keys(c)[0] === code)?.[code] || "")}>
                      {code} {renderPageBadges(selectedVisit["Page Numbers That Contain The Following Information"][3]["ICD10CM"].find(c => Object.keys(c)[0] === code)?.[code] || "")}
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="font-semibold">CPT Codes:</p>
                <ul>
                  {selectedVisit.General[0]["Medically Relevant Information That Is Present On All Pages"][0]["CPT Codes"].map((code, index) => (
                    <li key={index} className="cursor-pointer text-[#0084ff]" onClick={() => handleItemClick("CPT", code, selectedVisit["Page Numbers That Contain The Following Information"][4]["CPT Codes"].find(c => Object.keys(c)[0] === code)?.[code] || "")}>
                      {code} {renderPageBadges(selectedVisit["Page Numbers That Contain The Following Information"][4]["CPT Codes"].find(c => Object.keys(c)[0] === code)?.[code] || "")}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            <div className="mb-4">
              <p className="font-semibold">Prescriptions:</p>
              {selectedVisit.General[0]["Medically Relevant Information That Is Present On All Pages"][0]["Rx"].length > 0 ? (
                <ul>
                  {selectedVisit.General[0]["Medically Relevant Information That Is Present On All Pages"][0]["Rx"].map((rx, index) => (
                    <li key={index} className="cursor-pointer text-[#0084ff]" onClick={() => handleItemClick("Rx", rx, selectedVisit["Page Numbers That Contain The Following Information"][5]["Rx"].find(r => Object.keys(r)[0] === rx)?.[rx] || "")}>
                      {rx} {renderPageBadges(selectedVisit["Page Numbers That Contain The Following Information"][5]["Rx"].find(r => Object.keys(r)[0] === rx)?.[rx] || "")}
                    </li>
                  ))}
                </ul>
              ) : (
                <p>No prescriptions for this visit.</p>
              )}
            </div>
            <div className="mb-4">
              <p className="font-semibold">Other Medically Relevant Information:</p>
              <ul>
                {selectedVisit.General[0]["Medically Relevant Information That Is Present On All Pages"][0]["Other Medically Relevant Information"].map((info, index) =>
 (
                  <li key={index} className="cursor-pointer text-[#0084ff]" onClick={() => handleItemClick("Other", info, selectedVisit["Page Numbers That Contain The Following Information"][6]["Other Medically Relevant Information"].find(i => Object.keys(i)[0] === info)?.[info] || "")}>
                    {info} {renderPageBadges(selectedVisit["Page Numbers That Contain The Following Information"][6]["Other Medically Relevant Information"].find(i => Object.keys(i)[0] === info)?.[info] || "")}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <p className="font-semibold flex items-center"><FileText className="mr-2" size={16} /> Daily Summary:</p>
              <p>{selectedVisit.General[0]["Medically Relevant Information That Is Present On All Pages"][0]["Daily Summary"]}</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Right sidebar - Document Viewer */}
      <div className="w-1/4 bg-white p-4 border-l">
        <h2 className="text-xl font-bold mb-4">Document Viewer</h2>
        {initialVisitSelected ? (
          <div>
            <div className="mt-4 bg-gray-200 h-96 flex items-center justify-center">
              <p>PDF Viewer Placeholder</p>
            </div>
            <p className="mt-2 text-center">Page 1</p>
          </div>
        ) : selectedItem ? (
          <div>
            <p className="font-semibold">{selectedItem.type}: {selectedItem.value}</p>
            <p>Page(s): {selectedItem.pages}</p>
            <div className="mt-4 bg-gray-200 h-96 flex items-center justify-center">
              <p>PDF Viewer Placeholder</p>
            </div>
            <div className="flex justify-between mt-4">
              <Button onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))} disabled={currentPage === 1}>
                <ChevronLeft size={16} />
                Previous
              </Button>
              <span>Page {currentPage} of {selectedItem.pages.split(',').length}</span>
              <Button onClick={() => setCurrentPage(prev => Math.min(selectedItem.pages.split(',').length, prev + 1))} disabled={currentPage === selectedItem.pages.split(',').length}>
                Next
                <ChevronRight size={16} />
              </Button>
            </div>
          </div>
        ) : (
          <p>Select an item to view details</p>
        )}
      </div>
    </div>
  )
}