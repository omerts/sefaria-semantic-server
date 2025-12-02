'use client';

import { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';

interface SearchBarProps {
  onSearch: (query: string, bookPath?: string) => Promise<any>;
  searchQuery: string;
  setSearchQuery: (query: string) => void;
}

export default function SearchBar({
  onSearch,
  searchQuery,
  setSearchQuery,
}: SearchBarProps) {
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<any>(null);

  const handleSearch = async () => {
    if (!searchQuery.trim() || isSearching) return;

    try {
      setIsSearching(true);
      const searchResults = await onSearch(searchQuery);
      setResults(searchResults);
    } catch (err) {
      console.error('Search error:', err);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
      <div className="flex gap-3">
        <div className="flex-1 relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="חפש בכל הספרים..."
            className="w-full px-4 py-3 pr-12 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <Search className="absolute right-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
        </div>
        <button
          onClick={handleSearch}
          disabled={isSearching || !searchQuery.trim()}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white rounded-lg transition-colors flex items-center gap-2"
        >
          {isSearching ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              מחפש...
            </>
          ) : (
            <>
              <Search className="w-5 h-5" />
              חפש
            </>
          )}
        </button>
      </div>

      {results && (
        <div className="mt-4">
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
            נמצאו {results.total} תוצאות:
          </p>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {results.results.map((result: any, idx: number) => (
              <div
                key={idx}
                className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700"
              >
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">
                      {result.book} - {result.sefaria_ref}
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      {result.category.join(' > ')}
                    </p>
                  </div>
                  <span className="text-sm font-medium text-blue-600 dark:text-blue-400">
                    {result.score.toFixed(3)}
                  </span>
                </div>
                <p className="text-gray-700 dark:text-gray-300 mt-2">
                  {result.text}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


